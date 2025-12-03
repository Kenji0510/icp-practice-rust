use std::{collections::VecDeque, fs::File, io::BufReader, usize};

use anyhow::{Result, Context};
use icp_practice::{file_handler::load_pcd_files, operate_pcd::{PointXYZ, PointXYZT, Points, load_pcd_xyzt}, voxelization::voxel_downsample_array2};
use nalgebra::{Matrix3, Rotation3, SymmetricEigen, UnitQuaternion, Vector3};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
// use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{Inverse, Norm, SVD, Solve};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;
// use rayon::prelude::*;

// const WIDTH: f64 = 10.0;
// const HEIGHT: f64 = 5.0;
// const ROTATION_ANGLE_DEG: f64 = 25.0;
// const TRANSLATION_X: f64 = 5.0;
// const TRANSLATION_Y: f64 = 3.0;
// const NOISE_LEVEL: f64 = 0.1; // ノイズを少し強めに
const SAMPLE_SIZE: usize = 500;
const TRIM_PERCENTAGE: f64 = 1.0;
const K_NEIGHBORS: usize = 20;
const MAX_ITERATIONS: usize = 20;
const TOLERANCE: f32 = 0.050;  // Prev: 0.015
const VOXEL_SIZE: f32 = 0.2;

fn main() -> Result<()> {
    let target_pcd_dir = "data/input/mid360/pcd/mid360-20251125-07";
    let pcd_paths = match load_pcd_files(target_pcd_dir) {
        Ok(paths) => paths,
        Err(e) => {
            eprintln!("Error loading PCD files: {}", e);
            return Err(e);
        }
    };
    println!("Found {} PCD files in {}", pcd_paths.len(), target_pcd_dir);

    println!("Loading IMU JSON...");
    let imu_samples = load_and_flatten_imu_json("data/input/mid360/imu/mid360-imu-20251125-07/imu_data.json")
    // let imu_samples = load_imu_json("data/input/mid360/imu/mid360-imu-20251125-03/imu_data.json")
        .context("Failed to load IMU JSON data")?;
    println!("Loaded {} IMU samples.", imu_samples.len());

    if imu_samples.is_empty() {
        return Err(anyhow::anyhow!("IMU data is empty"));
    }
    
    let base_timestamp = imu_samples[0].timestamp_sec;
    println!("Base timestamp set to: {:.3}", base_timestamp);

    // let initial_pcd = match load_pcd_xyz(pcd_paths[0].to_str().unwrap()) {
    let initial_pcd = match load_pcd_xyzt(pcd_paths[0].to_str().unwrap()) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading initial PCD file: {}", e);
            return Err(e);
        }
    };

    let initial_pts_arr = point_xyzt_to_array2(&initial_pcd);
    let mut target_pts_arr = initial_pts_arr.clone();

    let mut current_global_pose = Array2::<f32>::eye(4);
    let mut last_delta_transform = Array2::<f32>::eye(4);

    // Local map queue
    let mut local_map_queue: VecDeque<Array2<f32>> = VecDeque::new();
    const LOCAL_MAP_SIZE: usize = 10;

    // Global map accumulator
    let mut global_map_accumulator: Vec<Array2<f32>> = Vec::new();
    global_map_accumulator.push(initial_pts_arr.clone());

    let mut final_errors: f32 = 0.0;

    // Time accumulators
    let mut total_preprocess_time = std::time::Duration::new(0, 0);
    let mut total_kdtree_time = std::time::Duration::new(0, 0);
    let mut total_normals_time = std::time::Duration::new(0, 0);
    let mut total_icp_time = std::time::Duration::new(0, 0);
    let mut processed_frame_count: u32 = 0;
    
    local_map_queue.push_back(initial_pts_arr.clone());
    global_map_accumulator.push(initial_pts_arr.clone());
    

    for (i, pcd_path) in pcd_paths.iter().enumerate() {
        println!("PCD File {}: {}", i, pcd_path.display());
        if i == 0 {
            continue;
        }

        // Loading current frame pcd
        // let current_pcd = match load_pcd_xyz(pcd_path.to_str().unwrap()) {
        let current_pcd = match load_pcd_xyzt(pcd_path.to_str().unwrap()) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Error loading PCD file {}: {}", pcd_path.display(), e);
                continue;
            }
        };

        if current_pcd.is_empty() { continue; }

        let start_time = std::time::Instant::now();
        
        // 点群の中から最小・最大時刻を探す（ソートされていなくても動くように）
        let min_timestamp = current_pcd.iter()
            .map(|p| p.timestamp).fold(f64::INFINITY, f64::min);
        let max_timestamp = current_pcd.iter()
            .map(|p| p.timestamp).fold(f64::NEG_INFINITY, f64::max);

        // 3. ★IMU軌跡 (Trajectory) の生成
        // その時刻範囲に対応するIMUデータを積分する
        let trajectory = build_rotation_trajectory(
            &imu_samples, 
            min_timestamp, 
            max_timestamp, 
        );

        // Preprocess for current points: deskew and range filter
        let mut current_pts_arr = preprocess_point_cloud(
            // &current_pts,
            &current_pcd,
            &trajectory,
            0.05 as f32,
            20.0 as f32,
        );
        let elapsed_preprocess = start_time.elapsed();

        current_pts_arr = voxel_downsample_array2(&current_pts_arr, VOXEL_SIZE);

        // Concatenate local map points
        let local_map_views: Vec<_> = local_map_queue.iter()
            .map(|p| p.view())
            .collect();
        target_pts_arr = ndarray::concatenate(
            Axis(0),
            &local_map_views
        ).context("Failed to concatenate arrays for local map")?;

        target_pts_arr = voxel_downsample_array2(&target_pts_arr, VOXEL_SIZE);

        // Create KdTree for target points
        println!("Building k-d tree for target points...");
        let target_points: Vec<[f32; 3]> = target_pts_arr.outer_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();
        let kdtree_target: kiddo::ImmutableKdTree<f32, 3> = kiddo::ImmutableKdTree::new_from_slice(&target_points);

        println!("k-d tree built with {} points.", target_pts_arr.nrows());
        let elapsed_kdtree = start_time.elapsed() - elapsed_preprocess;

        // Sourceの法線を計算 (数千点なので高速)
        let tx = current_global_pose[[0, 3]];
        let ty = current_global_pose[[1, 3]];
        let tz = current_global_pose[[2, 3]];

        let viewpoint = arr1(&[tx, ty, tz]); // ローカル座標系での視点
        let target_normals = calculate_normals_optimized(&target_pts_arr, &kdtree_target, &viewpoint)?;
        let elapsed_normals = start_time.elapsed() - elapsed_kdtree - elapsed_preprocess;

        let predicted_pose = last_delta_transform.dot(&current_global_pose);
        // ICPの探索開始位置を予測位置にセット
        let mut total_transform = predicted_pose;

        // Copy source points for current frame
        let source_points_num = current_pts_arr.nrows();
        let mut original_source_pts = Array2::<f32>::ones((source_points_num, 4));
        original_source_pts.slice_mut(s![.., 0..3]).assign(&current_pts_arr);

        let mut rng = thread_rng();
        let source_indices: Vec<usize> = (0..source_points_num).collect();

        let start_icp_time = std::time::Instant::now();
        for i in 0..MAX_ITERATIONS {
            // --- 2b. "現在" のソース点群を計算 ---
            let current_transformed_homogeneous = original_source_pts.dot(&total_transform.t());
            let current_source_pts_arr = current_transformed_homogeneous.slice(s![.., 0..3]).to_owned();

            let (sampled_source_pts, _) = 
                if source_points_num <= SAMPLE_SIZE {
                    (current_source_pts_arr.clone(), source_indices.clone())
                } else {
                    let indices = source_indices.as_slice()
                        .choose_multiple(&mut rng, SAMPLE_SIZE)
                        .cloned()
                        .collect::<Vec<usize>>();

                    (current_source_pts_arr.select(Axis(0), &indices), indices)
                };

            // --- 2c. `find_closest_pairs_kdtree` の呼び出し ---
            let (matched_target_pts, matched_target_indices, distance_sq) = 
                find_closest_pairs_kdtree(&sampled_source_pts, &target_pts_arr, &kdtree_target);

            // インライア選択
            let mut dist_with_indices: Vec<(f32, usize)> = distance_sq.iter()
                .cloned()
                .enumerate()
                .map(|(idx, dist)| (dist, idx))
                .collect();
            dist_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let n_to_keep = (dist_with_indices.len() as f64 * TRIM_PERCENTAGE) as usize;
            let inlier_indices: Vec<usize> = dist_with_indices.iter()
                .take(n_to_keep)
                .map(|&(_dist, idx)| idx)
                .collect();
            
            // インライアの点群を取得
            let inlier_source_pts = sampled_source_pts.select(Axis(0), &inlier_indices);
            let inlier_target_pts = matched_target_pts.select(Axis(0), &inlier_indices);

            // --- 2d. インライアの "法線" を取得 ---
            // `inlier_indices` を使って `matched_target_indices` から "グローバルインデックス" を取得
            let inlier_target_global_indices: Vec<usize> = inlier_indices.iter()
                .map(|&idx_n| matched_target_indices[idx_n])
                .collect();
            // グローバルインデックスを使って `target_normals` から法線を抽出
            let inlier_target_normals = target_normals.select(Axis(0), &inlier_target_global_indices);

            // --- 2e. "Point-to-Plane" ---
            let delta_transform = match calculate_transformation_pt_to_plane(
                &inlier_source_pts,
                &inlier_target_pts,
                &inlier_target_normals
            ) {
                Ok(tf) => tf,
                Err(e) => {
                    eprintln!("Warning: Failed to solve transformation, skipping iteration: {}", e);
                    continue;
                }
            };

            // --- 2f. "総" 変換行列を更新 ---
            // T_k+1 = DeltaT * T_k
            total_transform = delta_transform.dot(&total_transform);
            
            // --- 2g. エラー計算 (Point-to-Plane 誤差) ---
            let current_error = calculate_mean_pt_to_plane_error(
                &inlier_source_pts, 
                &inlier_target_pts, 
                &inlier_target_normals,
                &delta_transform // "今から" 適用する変換
            );
            
            // println!("Iteration {}: mean pt-to-plane error (from {} inliers, {:.0}% kept) = {}", 
            //     i + 1, 
            //     inlier_indices.len(), 
            //     TRIM_PERCENTAGE * 100.0,
            //     current_error
            // );

            final_errors = current_error;

            if current_error < TOLERANCE {
                println!("Converged at iteration {}", i + 1);
                println!("Final mean pt-to-plane error: {}", current_error);
                break;
            }
        }
        let elapsed_icp = start_icp_time.elapsed();

        println!("Final mean pt-to-plane error: {}", final_errors);

        let new_delta = total_transform.dot(&current_global_pose.inv().unwrap());

        // 移動量を計算 (回転行列のトレースから角度を、平行移動ベクトルから距離を算出)
        let translation_diff = new_delta.slice(s![0..3, 3]).norm(); // 移動距離 (m)
        // let trace = new_delta.diag().sum();
        let trace_3x3 = new_delta[[0, 0]] + new_delta[[1, 1]] + new_delta[[2, 2]];
        let cos_theta = ((trace_3x3 - 1.0) / 2.0).clamp(-1.0, 1.0);
        let rotation_diff = cos_theta.acos().abs(); // 回転角 (rad)

        last_delta_transform = new_delta;
        current_global_pose = total_transform.clone();

        //「一定以上動いた場合」 または 「最初の数フレーム」 だけマップ更新
        // これにより、停止時のノイズ蓄積を防ぎつつ、動いている時は滑らかに追従します
        const MOVE_THRESHOLD: f32 = 0.02; // 2cm以上動いたら
        const ANGLE_THRESHOLD: f32 = 0.035; // 約0.5度以上回ったら

        if i < 10 || translation_diff > MOVE_THRESHOLD {
            println!("Rotation diff: {:.4} rad, Translation diff: {:.4} m -- updating map", rotation_diff, translation_diff);
            
            if rotation_diff < ANGLE_THRESHOLD {
                // 1. 点群の変換
                let cloned_source_pts = original_source_pts.clone();
                let final_transformed_homogeneous = cloned_source_pts.dot(&total_transform.t());
                let aligned_pts = final_transformed_homogeneous.slice(s![.., 0..3]).to_owned();

                // 2. ローカルマップに追加
                if i % 2 == 0 {
                    local_map_queue.push_back(aligned_pts.clone());
                }
                
                if local_map_queue.len() > LOCAL_MAP_SIZE {
                    local_map_queue.pop_front();
                }

                // 3. グローバルマップへの保存
                if i % 5 == 0 {
                    global_map_accumulator.push(aligned_pts.to_owned());
                }
            }
        }

        // Debug
        // if i % 20 == 0 {
        //     // 最後に global_map_accumulator を全部結合して保存
        //     let final_map = ndarray::concatenate(Axis(0), &global_map_accumulator.iter().map(|a| a.view()).collect::<Vec<_>>())?;
        //     let voxelized_final_map = voxel_downsample_array2(&final_map, VOXEL_SIZE);
        //     let final_map_points = array2_to_points(&voxelized_final_map);
        //     let debug_save_path = format!("data/output/icp_map/debug/merged_until_{}.pcd", i);
        //     final_map_points.save_pcd(&debug_save_path, (0, 255, 0))
        //         .context("Failed to save debug merged PCD file")?;
        // }

        println!("Preprocessing time: {:.3?}, k-d tree time: {:.3?}, normals time: {:.3?}, ICP time: {:.3?}",
            elapsed_preprocess,
            elapsed_kdtree,
            elapsed_normals,
            elapsed_icp
        );

        total_preprocess_time += elapsed_preprocess;
        total_kdtree_time += elapsed_kdtree;
        total_normals_time += elapsed_normals;
        total_icp_time += elapsed_icp;
        processed_frame_count += 1;
    }

    if processed_frame_count > 0 {
        let avg_preprocess = total_preprocess_time / processed_frame_count;
        let avg_kdtree = total_kdtree_time / processed_frame_count;
        let avg_normals = total_normals_time / processed_frame_count;
        let avg_icp = total_icp_time / processed_frame_count;

        println!("\n--- Average Execution Times (over {} frames) ---", processed_frame_count);
        println!("Avg Preprocessing: {:.3?}", avg_preprocess);
        println!("Avg k-d tree:      {:.3?}", avg_kdtree);
        println!("Avg Normals:       {:.3?}", avg_normals);
        println!("Avg ICP:           {:.3?}", avg_icp);
    }

    println!("Current global pose:\n{}", current_global_pose);
    let tx = current_global_pose[[0, 3]];
    let ty = current_global_pose[[1, 3]];
    let tz = current_global_pose[[2, 3]];

    // 初期位置からの直線距離
    let distance_from_start = (tx*tx + ty*ty + tz*tz).sqrt();
    println!("Distance from start: {:.3} meters", distance_from_start);

    // Save final merged point cloud
    let final_points = array2_to_points(&target_pts_arr);
    let final_save_path = "data/output/icp_map/final_merged.pcd";
    final_points.save_pcd(final_save_path, (255, 0, 0))
        .context("Failed to save final merged PCD file")?;

    Ok(())
}

// (時刻, その時刻までの累積回転) のペア
type RotationTrajectory = Vec<(f64, UnitQuaternion<f64>)>;

/// 指定範囲のIMUデータを積分し、回転軌跡を作成 (バイアス補正付き)
fn build_rotation_trajectory(
    imu_samples: &[ImuSample], 
    start_time: f64,
    end_time: f64,
) -> RotationTrajectory {
    let mut trajectory = Vec::new();
    let mut current_rotation = UnitQuaternion::identity();
    
    // 範囲内のデータのみ抽出
    let buffer_time = 0.01; // 10ms余裕を持たせる
    let search_start = start_time - buffer_time;
    let search_end = end_time + buffer_time;

    let relevant_samples: Vec<&ImuSample> = imu_samples.iter()
        .filter(|s| s.timestamp_sec >= search_start && s.timestamp_sec <= search_end)
        .collect();

    // 最初の基準点
    trajectory.push((search_start, current_rotation));

    let mut last_time = search_start;

    for sample in relevant_samples {
        let dt = sample.timestamp_sec - last_time;

        if dt <= 1e-9 { 
            continue; 
        }

        // バイアスを引いて「真の回転」にする
        let wx = sample.gyro[0] as f64;
        let wy = sample.gyro[1] as f64;
        let wz = sample.gyro[2] as f64;
        let omega = Vector3::new(wx, wy, wz);

        // 微小回転を今の回転に積み上げる
        let angle_axis = omega * dt;
        let delta_q = UnitQuaternion::new(angle_axis);
        current_rotation = current_rotation * delta_q;

        // ★修正: 誤差蓄積を防ぐため正規化する
        current_rotation.renormalize();

        trajectory.push((sample.timestamp_sec, current_rotation));
        last_time = sample.timestamp_sec;
    }
    
    trajectory
}

/// 軌跡データから指定時刻の回転を球面線形補間(SLERP)して取得
fn get_rotation_at_time(traj: &RotationTrajectory, t: f64) -> UnitQuaternion<f64> {
    if traj.is_empty() { return UnitQuaternion::identity(); }
    if t <= traj.first().unwrap().0 { return traj.first().unwrap().1; }
    if t >= traj.last().unwrap().0 { return traj.last().unwrap().1; }

    // 線形探索
    for i in 0..traj.len()-1 {
        let (t0, q0) = traj[i];
        let (t1, q1) = traj[i+1];
        
        if t >= t0 && t <= t1 {
            let denom = t1 - t0;
            if denom.abs() < 1e-9 {
                return q0;
            }

            let ratio = (t - t0) / denom;
            
            // ratioが NaN になっていないか念のためチェック（デバッグ用）
            // if ratio.is_nan() {
            //      eprintln!("Error: NaN ratio detected at t={}, t0={}, t1={}", t, t0, t1);
            //      return q0;
            // }
            
            return q0.slerp(&q1, ratio);
        }
    }
    traj.last().unwrap().1
}

fn preprocess_point_cloud(
    points: &[PointXYZT],
    trajectory: &RotationTrajectory,
    min_dist: f32,
    max_dist: f32,
) -> Array2<f32> {
    let n_points = points.len();
    let mut valid_points_flat = Vec::with_capacity(n_points * 3);

    // 時間オフセットの調整用
    let time_offset = 0.0; 

    for p in points {
        let x = p.x as f32;
        let y = p.y as f32;
        let z = p.z as f32;

        // if x.is_nan() || y.is_nan() || z.is_nan() {
        //     println!("Warning: Found NaN point, skipping.");
        //     continue;
        // }

        // 1. 距離フィルタ
        let dist_sq = x * x + y * y + z * z;
        if dist_sq < min_dist * min_dist || dist_sq > max_dist * max_dist {
            continue;
        }

        // 2. 歪み補正 (Deskewing)
        // 点群が持っている正確な時刻を使用
        let point_time = p.timestamp + time_offset;

        // その時刻の回転姿勢を取得 (SLERP補間)
        let rotation = get_rotation_at_time(trajectory, point_time);

        // 座標変換 (逆回転させて開始時点の姿勢に戻す)
        let p_vec = Vector3::new(x as f64, y as f64, z as f64);
        
        // Livoxの場合、スキャン中に動いた分をキャンセル
        let corrected = rotation.inverse() * p_vec;

        // ★修正: 補正後の値が NaN になっていないかチェック
        if corrected.x.is_nan() || corrected.y.is_nan() || corrected.z.is_nan() {
            eprintln!("Warning: Deskew resulted in NaN for point ({}, {}, {}), skipping.", x, y, z);
            // Deskew計算でNaNが出た場合はスキップ
            continue;
        }

        // 3. データの格納
        valid_points_flat.push(corrected.x as f32);
        valid_points_flat.push(corrected.y as f32);
        valid_points_flat.push(corrected.z as f32);
    }

    let n_valid = valid_points_flat.len() / 3;
    Array2::from_shape_vec((n_valid, 3), valid_points_flat)
        .expect("Failed to create Array2 from valid points")
}

/// Point-to-Plane の平均二乗誤差 (RMSE) を計算する
fn calculate_mean_pt_to_plane_error(
    source_pts: &Array2<f32>, // 適用 "前" のソース点
    target_pts: &Array2<f32>,
    target_normals: &Array2<f32>,
    delta_transform: &Array2<f32> // "今から" 適用する微小変換
) -> f32 {
    
    let n = source_pts.nrows();
    
    // ソース点を同次座標系 (N, 4) に
    let mut source_homogeneous = Array2::<f32>::ones((n, 4));
    source_homogeneous.slice_mut(s![.., 0..3]).assign(source_pts);

    // 微小変換を適用
    let transformed_homogeneous = source_homogeneous.dot(&delta_transform.t());
    let transformed_pts = transformed_homogeneous.slice(s![.., 0..3]);

    // (target - transformed_source) . normal
    let diff = target_pts - &transformed_pts;
    
    // `diff` (N, 3) と `target_normals` (N, 3) の
    // 各行どうしの内積 (ドット積) を計算
    let errors = (&diff * target_normals) // 要素ごとの積
        .sum_axis(Axis(1))     // 行ごとに合計 (＝内積)
        .mapv(|val| val * val); // 2乗する

    (errors.sum() / n as f32).sqrt() // 二乗平均平方根 (RMSE)
}

fn calculate_transformation_pt_to_plane(
    inlier_source_pts: &Array2<f32>, // 現在のイテレーションのソース点 (N x 3)
    inlier_target_pts: &Array2<f32>, // 対応するターゲット点 (N x 3)
    inlier_target_normals: &Array2<f32> // 対応するターゲット法線 (N x 3)
) -> Result<Array2<f32>> { // 4x4 の "微小" 変換行列 (Delta T) を返す

    let n_inliers = inlier_source_pts.nrows();
    
    // A (ヤコビアン) は N x 6 の行列
    let mut a = Array2::<f32>::zeros((n_inliers, 6));
    // b (誤差) は N x 1 のベクトル
    let mut b = Array1::<f32>::zeros(n_inliers);

    azip!((
        mut a_row in a.axis_iter_mut(Axis(0)),
        b_val in &mut b,
        p_s in inlier_source_pts.axis_iter(Axis(0)), // p'_i (transformed source)
        p_t in inlier_target_pts.axis_iter(Axis(0)), // x_i (target)
        n_t in inlier_target_normals.axis_iter(Axis(0)) // n_i (target normal)
    ) { 
        // p'_i x n_i (クロス積)
        // Manual cross product: p'_i x n_i
        let cross_x = p_s[1] * n_t[2] - p_s[2] * n_t[1];
        let cross_y = p_s[2] * n_t[0] - p_s[0] * n_t[2];
        let cross_z = p_s[0] * n_t[1] - p_s[1] * n_t[0];

        // A の行 (1 x 6 ヤコビアン)
        a_row[0] = cross_x;
        a_row[1] = cross_y;
        a_row[2] = cross_z;
        a_row[3] = n_t[0];        // (n_i)_x
        a_row[4] = n_t[1];        // (n_i)_y
        a_row[5] = n_t[2];        // (n_i)_z
        
        // b の値 (スカラー)
        // b_i = (x_i - p'_i) . n_i
        let diff = &p_t - &p_s;
        *b_val = diff.dot(&n_t);
    });
    
    // (A^T A) x = (A^T b) の形で解く
    // at_a は (6xN) * (Nx6) = (6x6)
    let at_a = a.t().dot(&a);
    // at_b は (6xN) * (Nx1) = (6x1)
    let at_b = a.t().dot(&b);

    // 6x6 の小さな連立方程式を解く
    // x = (A^T A)^-1 * (A^T b)
    let x = at_a.solve(&at_b)
    .map_err(|e| anyhow::anyhow!("Linear solve failed: {:?}", e))
    .or_else(|_|  -> Result<Array1<f32>> {
        // もし A^T A が特異行列 (解けない) なら、SVDで擬似逆行列を使って解く
        // (これはロバスト性のためのフォールバック)
        println!("Warning: Falling back to SVD solver for linear system.");
        let (u, s, vt) = at_a.svd(true, true)
            .map_err(|e| anyhow::anyhow!("SVD failed: {:?}", e))?;
        
        let u = u.context("SVD failed: U matrix is None")?;
        let vt = vt.context("SVD failed: Vt matrix is None")?;
        
        let s_inv = s.mapv(|v| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 });
        let pinv = vt.t().dot(&Array2::from_diag(&s_inv)).dot(&u.t());
        Ok(pinv.dot(&at_b))
    })?;

    // x (6x1 ベクトル) を 4x4 の微小変換行列 Delta T に変換
    let alpha = x[0];
    let beta = x[1];
    let gamma = x[2];
    let tx = x[3];
    let ty = x[4];
    let tz = x[5];

    // 1. 回転ベクトル [α, β, γ] から「厳密な」3x3回転行列 R を計算
    let theta = (alpha*alpha + beta*beta + gamma*gamma).sqrt();
    let r: Array2<f32>; // 3x3 回転行列 R

    if theta < 1e-9 {
        // theta がほぼゼロなら、小角度近似（元の行列）でも安全
        r = array![
            [1.0, -gamma, beta],
            [gamma, 1.0, -alpha],
            [-beta, alpha, 1.0]
        ];
    } else {
        // Rodrigues' Rotation Formula (ロドリゲスの回転公式)
        let k_x = alpha / theta;
        let k_y = beta / theta;
        let k_z = gamma / theta;
        
        let c_th = theta.cos();
        let s_th = theta.sin();
        let v_th = 1.0 - c_th;

        // K (クロス積行列)
        // [ 0, -k_z,  k_y],
        // [ k_z,  0, -k_x],
        // [-k_y, k_x,   0]

        // R = I + sin(θ)K + (1-cos(θ))K^2
        r = array![
            [k_x*k_x*v_th + c_th,       k_x*k_y*v_th - k_z*s_th, k_x*k_z*v_th + k_y*s_th],
            [k_x*k_y*v_th + k_z*s_th, k_y*k_y*v_th + c_th,       k_y*k_z*v_th - k_x*s_th],
            [k_x*k_z*v_th - k_y*s_th, k_y*k_z*v_th + k_x*s_th, k_z*k_z*v_th + c_th]
        ];
    }
    
    // 2. 厳密な 4x4 剛体変換行列を構築
    let delta_t = array![
        [r[[0, 0]], r[[0, 1]], r[[0, 2]], tx],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]], ty],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]], tz],
        [0.0,       0.0,       0.0,       1.0]
    ];
    
    Ok(delta_t)
}

fn calculate_normals_optimized(
    target_pts: &Array2<f32>,
    kdtree: &kiddo::ImmutableKdTree<f32, 3>,
    viewpoint: &Array1<f32>,
) -> Result<Array2<f32>> {
    let n_points = target_pts.nrows();
    
    let normals_vec: Vec<Vec<f32>> = (0..n_points).into_par_iter().map(|i| {
        // 1. Query Point の取得
        let qx = target_pts[[i, 0]];
        let qy = target_pts[[i, 1]];
        let qz = target_pts[[i, 2]];
        let query_point = [qx, qy, qz];

        // 2. 近傍探索
        let k = std::num::NonZeroUsize::new(K_NEIGHBORS).unwrap();
        let neighbors = kdtree.nearest_n::<kiddo::SquaredEuclidean>(&query_point, k);

        if neighbors.len() < 3 {
            return vec![0.0, 0.0, 0.0];
        }

        // 3. 共分散行列の計算     
        // --- パス1: 重心 (Centroid) 計算 ---
        let mut sum = Vector3::zeros();
        for neighbor in &neighbors {
            let idx = neighbor.item as usize;
            let nx = target_pts[[idx, 0]];
            let ny = target_pts[[idx, 1]];
            let nz = target_pts[[idx, 2]];
            sum += Vector3::new(nx, ny, nz);
        }
        let k_f32 = neighbors.len() as f32;
        let centroid = sum / k_f32;

        // --- パス2: 共分散行列 (Covariance Matrix) 計算 ---
        let mut cov = Matrix3::zeros();
        for neighbor in &neighbors {
            let idx = neighbor.item as usize;
            let nx = target_pts[[idx, 0]];
            let ny = target_pts[[idx, 1]];
            let nz = target_pts[[idx, 2]];
            
            let d = Vector3::new(nx, ny, nz) - centroid;
            cov += d * d.transpose();
        }

        // 4. 固有値分解 (Symmetric Eigen decomposition)
        let eigen = SymmetricEigen::new(cov);
        
        let (min_idx, _) = eigen.eigenvalues.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // そのインデックスに対応する固有ベクトルを取得
        let mut normal = eigen.eigenvectors.column(min_idx).into_owned();

        // 法線の正規化
        let norm = normal.norm();
        if norm < 1e-12 {
            return vec![0.0, 0.0, 0.0];
        }
        normal /= norm;

        // 5. 視点方向への向き統一
        let view_dir = Vector3::new(viewpoint[0], viewpoint[1], viewpoint[2]) - centroid;
        if normal.dot(&view_dir) < 0.0 {
            normal = -normal;
        }

        vec![normal.x, normal.y, normal.z]
    }).collect();

    let mut normals_arr = Array2::<f32>::zeros((n_points, 3));
    for (i, normal) in normals_vec.into_iter().enumerate() {
        normals_arr[[i, 0]] = normal[0];
        normals_arr[[i, 1]] = normal[1];
        normals_arr[[i, 2]] = normal[2];
    }

    Ok(normals_arr)
}

fn array2_to_points(
    arr: &Array2<f32>
) -> Points {
    let mut pts = Vec::with_capacity(arr.len());
    for i in 0..arr.nrows() {
        pts.push(PointXYZ {
            x: arr[[i, 0]] as f32,
            y: arr[[i, 1]] as f32,
            z: arr[[i, 2]] as f32,
        });
    }

    Points::new(pts)
}

fn points_to_array2(
    points: &Points
) -> Array2<f32> {
    let n = points.points.len();
    let mut arr = Array2::<f32>::zeros((n, 3));
    for (i, p) in points.points.iter().enumerate() {
        arr[[i, 0]] = p.x as f32;
        arr[[i, 1]] = p.y as f32;
        arr[[i, 2]] = p.z as f32;
    }

    arr
}

fn point_xyzt_to_array2(
    points: &[PointXYZT]
) -> Array2<f32> {
    let n = points.len();
    let mut arr = Array2::<f32>::zeros((n, 3));
    for (i, p) in points.iter().enumerate() {
        arr[[i, 0]] = p.x as f32;
        arr[[i, 1]] = p.y as f32;
        arr[[i, 2]] = p.z as f32;
    }

    arr
}

fn find_closest_pairs_kdtree(
    source_pts: &Array2<f32>,      // サンプリングされた source 点群
    target_pts: &Array2<f32>,
    kdtree: &kiddo::ImmutableKdTree<f32, 3>
) -> (Array2<f32>, Vec<usize>, Vec<f32>) {
    
    let n = source_pts.nrows();

    let results: Vec<(usize, f32)> = (0..n).into_par_iter()
        .filter_map(|i| {
            let source_row = source_pts.row(i);
            let query_point = [source_row[0], source_row[1], source_row[2]];

            let k = std::num::NonZeroUsize::new(1).unwrap();
            let neighbors = kdtree.nearest_n::<kiddo::SquaredEuclidean>(&query_point, k);

            let neighbor = neighbors[0];
            let dist_sq = neighbor.distance;
            let target_index = neighbor.item as usize;
            Some((target_index, dist_sq))
        })
        .collect();

    let (closest_indices, distance_sq): (Vec<usize>, Vec<f32>) = results.into_iter().unzip();
    
    let matched_target_pts = target_pts.select(Axis(0), &closest_indices);
    (matched_target_pts, closest_indices, distance_sq)
}

// JSONの構造に合わせた定義
#[derive(Debug, Deserialize)]
struct LivoxImuBatch {
    timestamp: u64, // ナノ秒と仮定 (例: 484350964530)
    angular_velocity: Vec<[f32; 3]>,
    sample_count: usize,
    // sample_count は Vecのlen()でわかるので無視してもOK
}

// 扱いやすいように変換した後の1サンプルあたりの構造体
#[derive(Debug, Clone)]
struct ImuSample {
    timestamp_sec: f64, // 計算しやすいように秒単位(f64)に変換して持つ
    gyro: Array1<f32>,
    sample_count: usize,
}

/// JSONファイルを読み込み、時間順に並んだサンプルのリストを返す
fn load_and_flatten_imu_json(path: &str) -> Result<Vec<ImuSample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let batches: Vec<LivoxImuBatch> = serde_json::from_reader(reader)?;

    let mut samples = Vec::new();
    let interval_sec = 0.005; // 200Hz = 5ms

    for batch in batches {
        // バッチの基準時刻 (u64ナノ秒 -> f64秒 に変換)
        // ※もしJSONのtimestampがマイクロ秒やミリ秒なら、ここの割り算を変えてください
        let start_time_sec = batch.timestamp as f64 / 1_000_000_000.0;

        for (i, gyro) in batch.angular_velocity.iter().enumerate() {
            // 各サンプルの時刻を計算
            let current_time = start_time_sec + (i as f64 * interval_sec);
            
            samples.push(ImuSample {
                timestamp_sec: current_time,
                gyro: arr1(&[gyro[0] as f32, gyro[1] as f32, gyro[2] as f32]),
                sample_count: 1,
            });
        }
    }

    // 念のため時間順にソート（JSONが順番通りなら不要だが安全のため）
    samples.sort_by(|a, b| a.timestamp_sec.partial_cmp(&b.timestamp_sec).unwrap());

    Ok(samples)
}