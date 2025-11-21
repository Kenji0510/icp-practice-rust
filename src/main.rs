use std::{collections::VecDeque, fs::File, io::BufReader};

use anyhow::{Result, Context};
use icp_practice::{file_handler::load_pcd_files, operate_pcd::{PointXYZ, PointXYZNormal, Points, load_pcd_xyz, save_pcd, save_pcd_with_normals}, voxelization::voxel_downsample_array2};
use kdtree::{KdTree, distance::squared_euclidean};
use nalgebra::{Matrix3, Matrix6, Rotation3, SymmetricEigen, Vector3, Vector6};
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
const SAMPLE_SIZE: usize = 300;
const TRIM_PERCENTAGE: f64 = 1.0;
const K_NEIGHBORS: usize = 15;
const MAX_ITERATIONS: usize = 15;
const TOLERANCE: f64 = 0.013;  // Prev: 0.015
const VOXEL_SIZE: f64 = 0.2;

fn main() -> Result<()> {
    let scan_interval = 0.1; // 10Hz = 0.1秒間隔
    let target_pcd_dir = "data/input/4201/voxel-005";
    let pcd_paths = match load_pcd_files(target_pcd_dir) {
        Ok(paths) => paths,
        Err(e) => {
            eprintln!("Error loading PCD files: {}", e);
            return Err(e);
        }
    };
    println!("Found {} PCD files in {}", pcd_paths.len(), target_pcd_dir);
    // for path in &pcd_paths {
    //     println!(" - {}", path.display());
    // }

    println!("Loading IMU JSON...");
    let imu_samples = load_and_flatten_imu_json("data/input/imu-json/imu_data.json")
        .context("Failed to load IMU JSON data")?;
    println!("Loaded {} IMU samples.", imu_samples.len());

    if imu_samples.is_empty() {
        return Err(anyhow::anyhow!("IMU data is empty"));
    }
    
    // ★基準時刻の設定: IMUデータの最初の時間をスタート地点(T0)とする
    // もし「PCDの方が5秒遅れて始まる」等のオフセットがある場合はここで調整してください
    let base_timestamp = imu_samples[0].timestamp_sec;
    println!("Base timestamp set to: {:.3}", base_timestamp);

    let initial_pcd = match load_pcd_xyz(pcd_paths[0].to_str().unwrap()) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading initial PCD file: {}", e);
            return Err(e);
        }
    };

    let initial_pts = Points::new(initial_pcd);
    let initial_pts_arr = points_to_array2(&initial_pts);
    let mut target_pts_arr = initial_pts_arr.clone();

    let mut current_global_pose = Array2::<f64>::eye(4);
    let mut last_delta_transform = Array2::<f64>::eye(4); // 直近の速度（移動量）を保持

    // Local map queue
    let mut local_map_queue: VecDeque<Array2<f64>> = VecDeque::new();
    const LOCAL_MAP_SIZE: usize = 10;

    // Global map accumulator
    let mut global_map_accumulator: Vec<Array2<f64>> = Vec::new();
    global_map_accumulator.push(initial_pts_arr.clone());

    // 初期フレームの法線を計算してキューに入れる処理
    {
        // 初期フレーム用のKdTreeと法線計算
        let mut kdtree_init: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);
        // ... (add points to kdtree) ...
        for (i, r) in initial_pts_arr.rows().into_iter().enumerate() {
             let point: [f64; 3] = [r[0], r[1], r[2]];
             kdtree_init.add(point, i).unwrap();
        }
        let viewpoint = arr1(&[0.0, 0.0, 0.0]);
        // let initial_normals = calculate_normals(&initial_pts_arr, &kdtree_init, &viewpoint)?;
        
        local_map_queue.push_back(initial_pts_arr.clone());
        global_map_accumulator.push(initial_pts_arr.clone());
    }

    for (i, pcd_path) in pcd_paths.iter().enumerate() {
        println!("PCD File {}: {}", i, pcd_path.display());
        if i == 0 {
            continue;
        }

        // Loading current frame pcd
        let current_pcd = match load_pcd_xyz(pcd_path.to_str().unwrap()) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Error loading PCD file {}: {}", pcd_path.display(), e);
                continue;
            }
        };

        let start_time = std::time::Instant::now();
        let current_pts = Points::new(current_pcd);
        // let current_pts_arr = points_to_array2(&current_pts);

        // // このフレームの開始時刻 = 基準時刻 + (インデックス * 0.1秒)
        let current_frame_start_time = base_timestamp + (i as f64 * scan_interval);

        // // 対応するIMUデータの平均角速度を取得
        let avg_gyro = get_avg_gyro(
            &imu_samples, 
            current_frame_start_time, 
            scan_interval
        );

        // // デバッグ表示: ちゃんと値が取れているか確認
        // // println!(" - Time: {:.3}s ~ {:.3}s, Gyro: {:.3?}", 
        // //     current_frame_start_time, 
        // //     current_frame_start_time + scan_interval, 
        // //     avg_gyro
        // // );

        // // 歪み補正 (Deskewing) 実行
        // let mut current_pts_arr = match avg_gyro {
        //     Some(gyro) => {
        //         deskew_point_cloud(&current_pts_arr, &gyro, scan_interval)
        //     }
        //     None => {
        //         eprintln!("Warning: No IMU data for frame {} time window [{:.3}, {:.3}), using original points",
        //             i, current_frame_start_time, current_frame_start_time + scan_interval);
        //         current_pts_arr  // Use original points if no IMU data
        //     }
        // };

        // // 2. ★ここで距離フィルタを実行★
        // // 例: 0.5m 以内(自分)と、30m 以遠(ノイズ)をカット
        // // 屋内なら 20.0〜30.0m、屋外でも 50.0m 程度で切るのが一般的
        // let current_pts_arr = filter_by_range(&current_pts_arr, 0.1, 20.0);

        // Preprocess for current points: deskew and range filter
        let mut current_pts_arr = preprocess_point_cloud(
            &current_pts,
            avg_gyro.as_ref(),
            scan_interval,
            0.05,
            15.0,
        );
        let elapsed_preprocess = start_time.elapsed();

        current_pts_arr = voxel_downsample_array2(&current_pts_arr, VOXEL_SIZE);

        // Concatenate local map points
        let local_map_views: Vec<_> = local_map_queue.iter()
            .map(|p| p.view())
            .collect();
        target_pts_arr = ndarray::concatenate(
            Axis(0),  // 縦方向（行方向）に結合
            &local_map_views
        ).context("Failed to concatenate arrays for local map")?;

        target_pts_arr = voxel_downsample_array2(&target_pts_arr, VOXEL_SIZE);

        // Create KdTree for target points
        println!("Building k-d tree for target points...");
        // let n_dims_target = target_pts_arr.ncols();
        let mut kdtree_target: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);

        for (i, point_row) in target_pts_arr.rows().into_iter().enumerate() {
            let point_slice = point_row.as_slice().unwrap();
            let point: [f64; 3] = [point_slice[0], point_slice[1], point_slice[2]];
            kdtree_target.add(point, i).unwrap();
        }
        println!("k-d tree built with {} points.", target_pts_arr.nrows());
        let elapsed_kdtree = start_time.elapsed() - elapsed_preprocess;

        // Concatenate normals for local map
        // let local_map_normals_views: Vec<_> = local_map_queue.iter().map(|(_, n)| n.view()).collect();
        // let target_normals = ndarray::concatenate(Axis(0), &local_map_normals_views)?;

        // Calculate normals for target points
        // let viewpoint: Array1<f64> = arr1(&[0.0, 0.0, 0.0]);
        // println!("Calculating normals for target points (k={})...", K_NEIGHBORS);

        // let target_normals = calculate_normals(&target_pts_arr, &kdtree, &viewpoint)
        //     .context("Failed to calculate normals")?;
        // let elapsed_normals = start_time.elapsed() - elapsed_kdtree - elapsed_preprocess;

        // Current用のKdTree (法線計算のためだけに一時作成)
        let mut kdtree_current: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);
        for (i, point_row) in current_pts_arr.rows().into_iter().enumerate() {
            let point: [f64; 3] = [point_row[0], point_row[1], point_row[2]];
            kdtree_current.add(point, i).unwrap();
        }

        // Sourceの法線を計算 (数千点なので高速)
        let tx = current_global_pose[[0, 3]];
        let ty = current_global_pose[[1, 3]];
        let tz = current_global_pose[[2, 3]];
        let viewpoint = arr1(&[tx, ty, tz]); // ローカル座標系での視点
        // let current_normals_arr = calculate_normals(&current_pts_arr, &kdtree_current, &viewpoint)?;
        // let target_normals = calculate_normals(&target_pts_arr, &kdtree_target, &viewpoint)?;
        let target_normals = calculate_normals_optimized(&target_pts_arr, &kdtree_target, &viewpoint)?;
        let elapsed_normals = start_time.elapsed() - elapsed_kdtree - elapsed_preprocess;

        let predicted_pose = last_delta_transform.dot(&current_global_pose);
        // let mut total_transform = current_global_pose.clone();
        // ICPの探索開始位置を予測位置にセット
        let mut total_transform = predicted_pose;

        // Copy source points for current frame
        let source_points_num = current_pts_arr.nrows();
        let mut original_source_pts = Array2::<f64>::ones((source_points_num, 4));
        original_source_pts.slice_mut(s![.., 0..3]).assign(&current_pts_arr);

        let mut rng = thread_rng();
        let source_indices: Vec<usize> = (0..source_points_num).collect();

        let start_icp_time = std::time::Instant::now();
        for i in 0..MAX_ITERATIONS {
            // --- 2b. "現在" のソース点群を計算 ---
            // (N, 4) = (N, 4) .dot (4, 4)
            let current_transformed_homogeneous = original_source_pts.dot(&total_transform.t());
            // (N, 3) の座標に戻す
            let current_source_pts_arr = current_transformed_homogeneous.slice(s![.., 0..3]).to_owned();

            // (サンプリングは current_source_pts_arr から行う - 変更なし)
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

            // --- 2c. `find_closest_pairs_kdtree` の呼び出し (戻り値が3つに) ---
            let (matched_target_pts, matched_target_indices, distance_sq) = 
                find_closest_pairs_kdtree(&sampled_source_pts, &target_pts_arr, &kdtree_target);

            // (Trimming (インライア選択) - 変更なし)
            let mut dist_with_indices: Vec<(f64, usize)> = distance_sq.iter()
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
            
            // (インライアの点群を取得 - 変更なし)
            let inlier_source_pts = sampled_source_pts.select(Axis(0), &inlier_indices);
            let inlier_target_pts = matched_target_pts.select(Axis(0), &inlier_indices);

            // --- 2d. インライアの "法線" を取得 (★重要★) ---
            // `inlier_indices` を使って `matched_target_indices` から "グローバルインデックス" を取得
            let inlier_target_global_indices: Vec<usize> = inlier_indices.iter()
                .map(|&idx_n| matched_target_indices[idx_n]) // idx_n は 0..N_sample のインデックス
                .collect();
            // グローバルインデックスを使って `target_normals` から法線を抽出
            let inlier_target_normals = target_normals.select(Axis(0), &inlier_target_global_indices);

            // --- 2e. "Point-to-Plane" の計算を呼び出し ---
            let delta_transform = match calculate_transformation_pt_to_plane(
            // let delta_transform = match calculate_transformation_pt_to_plane_optimized(
                &inlier_source_pts,
                &inlier_target_pts,
                &inlier_target_normals
            ) {
                Ok(tf) => tf,
                Err(e) => {
                    eprintln!("Warning: Failed to solve transformation, skipping iteration: {}", e);
                    continue; // このイテレーションをスキップ
                }
            };

            // --- 2f. "総" 変換行列を更新 ---
            // T_k+1 = DeltaT * T_k
            total_transform = delta_transform.dot(&total_transform);
            
            // --- 2g. エラー計算 (Point-to-Plane 誤差を推奨) ---
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

            if current_error < TOLERANCE {
                println!("Converged at iteration {}", i + 1);
                println!("Final mean pt-to-plane error: {}", current_error);
                break;
            }
        }
        let elapsed_icp = start_icp_time.elapsed();

        let new_delta = total_transform.dot(&current_global_pose.inv().unwrap());

        // 移動量を計算 (回転行列のトレースから角度を、平行移動ベクトルから距離を算出)
        let translation_diff = new_delta.slice(s![0..3, 3]).norm(); // 移動距離 (m)
        let trace = new_delta.diag().sum();
        let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
        let rotation_diff = cos_theta.acos().abs(); // 回転角 (rad)

        last_delta_transform = new_delta;
        current_global_pose = total_transform.clone();

        // ★修正: 「一定以上動いた場合」 または 「最初の数フレーム」 だけマップ更新
        // これにより、停止時のノイズ蓄積を防ぎつつ、動いている時は滑らかに追従します
        const MOVE_THRESHOLD: f64 = 0.02; // 2cm以上動いたら
        const ANGLE_THRESHOLD: f64 = 0.5; // 約0.5度以上回ったら

        if i < 10 || translation_diff > MOVE_THRESHOLD || rotation_diff > ANGLE_THRESHOLD {
            // println!("Updating local map at frame {}", i);
            
            // 1. 点群の変換
            let cloned_source_pts = original_source_pts.clone();
            let final_transformed_homogeneous = cloned_source_pts.dot(&total_transform.t());
            let aligned_pts = final_transformed_homogeneous.slice(s![.., 0..3]).to_owned();

            // 2. 法線の回転
            let rotation_matrix = total_transform.slice(s![0..3, 0..3]);
            // let aligned_normals = current_normals_arr.dot(&rotation_matrix.t());

            // 3. ローカルマップに追加 (毎フレームに近い頻度で行われる)
            local_map_queue.push_back(aligned_pts.clone());
            
            if local_map_queue.len() > LOCAL_MAP_SIZE {
                local_map_queue.pop_front();
            }

            // 4. グローバルマップへの保存 (こちらは容量節約のため、たまにでOK)
            // ここは i % 5 のままで良いですし、上記と同じ移動判定を使っても良いです
            if i % 5 == 0 {
                global_map_accumulator.push(aligned_pts.to_owned());
            }
        }

        // Debug
        if i % 10 == 0 {
            // 最後に global_map_accumulator を全部結合して保存
            let final_map = ndarray::concatenate(Axis(0), &global_map_accumulator.iter().map(|a| a.view()).collect::<Vec<_>>())?;
            let voxelized_final_map = voxel_downsample_array2(&final_map, VOXEL_SIZE);
            let final_map_points = array2_to_points(&voxelized_final_map);
            let debug_save_path = format!("data/output/icp_map/debug/merged_until_{}.pcd", i);
            final_map_points.save_pcd(&debug_save_path, (0, 255, 0))
                .context("Failed to save debug merged PCD file")?;
        }

        println!("Preprocessing time: {:.3?}, k-d tree time: {:.3?}, normals time: {:.3?}, ICP time: {:.3?}",
            elapsed_preprocess,
            elapsed_kdtree,
            elapsed_normals,
            elapsed_icp
        );
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

fn preprocess_point_cloud(
    points: &Points,
    gyro: Option<&Array1<f64>>,
    scan_interval: f64,
    min_dist: f64,
    max_dist: f64,
) -> Array2<f64> {
    let n_points = points.points.len();

    let mut valid_points_flat = Vec::with_capacity(n_points * 3);

    // Angular velocities
    let (wx, wy, wz) = match gyro {
        Some(g) => (g[0], g[1], g[2]),
        None => (0.0, 0.0, 0.0),
    };

    for (i, p) in points.points.iter().enumerate() {
        let mut x = p.x as f64;
        let mut y = p.y as f64;
        let mut z = p.z as f64;

        // Filter by range
        let dist_sq = x * x + y * y + z * z;
        if dist_sq < min_dist * min_dist || dist_sq > max_dist * max_dist {
            continue;
        }

        // Deskewing
        // 2. 歪み補正 (Deskewing)
        // IMUデータが存在し、かつ角速度がほぼゼロでない場合のみ計算
        if gyro.is_some() && (wx.abs() > 1e-6 || wy.abs() > 1e-6 || wz.abs() > 1e-6) {
            let ratio = i as f64 / n_points as f64;
            let dt = ratio * scan_interval;

            // ロドリゲスの回転公式の簡易版（微小回転近似）
            // R ≈ I + [ω]_x * dt
            // これにより sin/cos の計算コストを削減できる（精度が必要なら正規のロドリゲスを使用）
            let dx = (wy * z - wz * y) * dt;
            let dy = (wz * x - wx * z) * dt;
            let dz = (wx * y - wy * x) * dt;

            x += dx;
            y += dy;
            z += dz;
        }

        // 3. データの格納
        valid_points_flat.push(x);
        valid_points_flat.push(y);
        valid_points_flat.push(z);
    }

    let n_valid = valid_points_flat.len() / 3;
    Array2::from_shape_vec((n_valid, 3), valid_points_flat)
        .expect("Failed to create Array2 from valid points")
}

/// Point-to-Plane の平均二乗誤差 (RMSE) を計算する
fn calculate_mean_pt_to_plane_error(
    source_pts: &Array2<f64>, // 適用 "前" のソース点
    target_pts: &Array2<f64>,
    target_normals: &Array2<f64>,
    delta_transform: &Array2<f64> // "今から" 適用する微小変換
) -> f64 {
    
    let n = source_pts.nrows();
    
    // ソース点を同次座標系 (N, 4) に
    let mut source_homogeneous = Array2::<f64>::ones((n, 4));
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

    (errors.sum() / n as f64).sqrt() // 二乗平均平方根 (RMSE)
}

fn calculate_transformation_pt_to_plane_optimized(
    inlier_source_pts: &Array2<f64>,
    inlier_target_pts: &Array2<f64>,
    inlier_target_normals: &Array2<f64>
) -> Result<Array2<f64>> {
    
    // Rayon による並列 Map-Reduce パターン
    // 各スレッドで 6x6 行列 (H) と 6x1 ベクトル (b) を部分的に計算して足し合わせる
    let (h, g) = (0..inlier_source_pts.nrows())
        .into_par_iter()
        .fold(
            || (Matrix6::zeros(), Vector6::zeros()), // 初期値 (各スレッドのローカル変数)
            |(mut acc_h, mut acc_g), i| {
                // データの読み出し (スタック変数へ)
                let s_x = inlier_source_pts[[i, 0]];
                let s_y = inlier_source_pts[[i, 1]];
                let s_z = inlier_source_pts[[i, 2]];
                let ps = Vector3::new(s_x, s_y, s_z);

                let t_x = inlier_target_pts[[i, 0]];
                let t_y = inlier_target_pts[[i, 1]];
                let t_z = inlier_target_pts[[i, 2]];
                let pt = Vector3::new(t_x, t_y, t_z);

                let n_x = inlier_target_normals[[i, 0]];
                let n_y = inlier_target_normals[[i, 1]];
                let n_z = inlier_target_normals[[i, 2]];
                let n = Vector3::new(n_x, n_y, n_z);

                // --- Jacobian (J) の計算 ---
                // J = [ (ps x n)^T,  n^T ]  (1行6列)
                
                // 1. 回転成分: ps と n の外積
                let cross = ps.cross(&n);

                // 2. 6次元ベクトル J_vec を作成
                // alpha, beta, gamma, tx, ty, tz の順に対応
                let j_vec = Vector6::new(
                    cross[0], cross[1], cross[2], // 回転成分
                    n[0],     n[1],     n[2]      // 平行移動成分
                );

                // --- 右辺 (Error) の計算 ---
                // r = (pt - ps) . n
                let diff = pt - ps;
                let residual = diff.dot(&n);

                // --- 累積 (Accumulation) ---
                // H += J^T * J  (6x6行列の加算)
                // G += J^T * r  (6x1ベクトルの加算)
                
                // nalgebra の syger (Rank-1 update) を使うとさらに速いが、
                // 単純な加算でもコンパイラが最適化してくれる
                acc_h += j_vec * j_vec.transpose();
                acc_g += j_vec * residual;

                (acc_h, acc_g)
            }
        )
        .reduce(
            || (Matrix6::zeros(), Vector6::zeros()), // Reduce時の初期値
            |(h1, g1), (h2, g2)| {
                (h1 + h2, g1 + g2) // 行列とベクトルの単純加算
            }
        );

    // --- 連立方程式 Hx = g を解く ---
    // 6x6 なので Cholesky分解 が高速
    let x = h.cholesky()
        .ok_or_else(|| anyhow::anyhow!("Cholesky decomposition failed (matrix not positive definite)"))?
        .solve(&g);

    // 結果のベクトル x = [alpha, beta, gamma, tx, ty, tz]
    let alpha = x[0];
    let beta = x[1];
    let gamma = x[2];
    let tx = x[3];
    let ty = x[4];
    let tz = x[5];

    // --- 回転行列の構築 (ここは以前と同じロジック) ---
    let theta_sq = alpha*alpha + beta*beta + gamma*gamma;
    let r_mat: nalgebra::Matrix3<f64>;

    if theta_sq < 1e-12 {
        // 微小回転近似
        r_mat = nalgebra::Matrix3::new(
             1.0, -gamma,  beta,
             gamma,  1.0, -alpha,
            -beta,  alpha,  1.0
        );
    } else {
        let theta = theta_sq.sqrt();
        let k = Vector3::new(alpha, beta, gamma) / theta;
        let k_cross = nalgebra::Matrix3::new(
            0.0, -k.z, k.y,
            k.z, 0.0, -k.x,
            -k.y, k.x, 0.0
        );
        // Rodrigues
        let i = nalgebra::Matrix3::identity();
        r_mat = i + k_cross * theta.sin() + (k_cross * k_cross) * (1.0 - theta.cos());
    }

    // --- Array2<f64> (4x4) に変換して返す ---
    let mut delta_t = Array2::<f64>::eye(4);
    
    // 回転成分コピー
    delta_t[[0,0]] = r_mat[(0,0)]; delta_t[[0,1]] = r_mat[(0,1)]; delta_t[[0,2]] = r_mat[(0,2)];
    delta_t[[1,0]] = r_mat[(1,0)]; delta_t[[1,1]] = r_mat[(1,1)]; delta_t[[1,2]] = r_mat[(1,2)];
    delta_t[[2,0]] = r_mat[(2,0)]; delta_t[[2,1]] = r_mat[(2,1)]; delta_t[[2,2]] = r_mat[(2,2)];
    
    // 平行移動成分
    delta_t[[0,3]] = tx;
    delta_t[[1,3]] = ty;
    delta_t[[2,3]] = tz;

    Ok(delta_t)
}

fn calculate_transformation_pt_to_plane(
    inlier_source_pts: &Array2<f64>, // 現在のイテレーションのソース点 (N x 3)
    inlier_target_pts: &Array2<f64>, // 対応するターゲット点 (N x 3)
    inlier_target_normals: &Array2<f64> // 対応するターゲット法線 (N x 3)
) -> Result<Array2<f64>> { // 4x4 の "微小" 変換行列 (Delta T) を返す

    let n_inliers = inlier_source_pts.nrows();
    
    // A (ヤコビアン) は N x 6 の行列
    let mut a = Array2::<f64>::zeros((n_inliers, 6));
    // b (誤差) は N x 1 のベクトル
    let mut b = Array1::<f64>::zeros(n_inliers);

    // azip! を使って並列に A と b を構築
    azip!((
        mut a_row in a.axis_iter_mut(Axis(0)),
        b_val in &mut b,
        p_s in inlier_source_pts.axis_iter(Axis(0)), // p'_i (transformed source)
        p_t in inlier_target_pts.axis_iter(Axis(0)), // x_i (target)
        n_t in inlier_target_normals.axis_iter(Axis(0)) // n_i (target normal)
    ) {
        // --- ここがPoint-to-Planeの核心 ---
        
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
    .or_else(|_|  -> Result<Array1<f64>> {
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
    let r: Array2<f64>; // 3x3 回転行列 R

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

    // 2. 並進ベクトル t (3x1) を計算 (これは回転の影響も受ける)
    // (ここでは簡略化のため、並進はそのまま t = [tx, ty, tz] とします。
    //  厳密な指数写像では V * t も計算しますが、ICPではこの形でも十分に収束します)
    
    // 3. 厳密な 4x4 剛体変換行列を構築
    let delta_t = array![
        [r[[0, 0]], r[[0, 1]], r[[0, 2]], tx],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]], ty],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]], tz],
        [0.0,       0.0,       0.0,       1.0]
    ];
    
    Ok(delta_t)
}

fn create_points_with_normals(
    points: &Array2<f64>,
    normals: &Array2<f64>,
) -> Vec<PointXYZNormal> {
    assert_eq!(points.nrows(), normals.nrows(), "Points and normals must have same number of rows");
    assert_eq!(points.ncols(), 3, "Points must be 3D");
    assert_eq!(normals.ncols(), 3, "Normals must be 3D");
    
    let n = points.nrows();
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        result.push(PointXYZNormal {
            x: points[[i, 0]] as f32,
            y: points[[i, 1]] as f32,
            z: points[[i, 2]] as f32,
            normal_x: normals[[i, 0]] as f32,
            normal_y: normals[[i, 1]] as f32,
            normal_z: normals[[i, 2]] as f32,
        });
    }
    
    result
}

fn calculate_normals_optimized(
    target_pts: &Array2<f64>,
    kdtree: &KdTree<f64, usize, [f64; 3]>,
    viewpoint: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_points = target_pts.nrows();
    
    // 結果を格納する配列 (スレッドセーフに書き込むため UnsafeCell あるいは Vec で collect する)
    // Rayonの map/collect を使うのが最も安全で高速です
    let normals_vec: Vec<Vec<f64>> = (0..n_points).into_par_iter().map(|i| {
        // 1. Query Point の取得
        // ndarrayの行アクセスは少し遅いので、生ポインタ的アクセスかgetを使う
        let qx = target_pts[[i, 0]];
        let qy = target_pts[[i, 1]];
        let qz = target_pts[[i, 2]];
        let query_point = [qx, qy, qz];

        // 2. 近傍探索
        let neighbors = match kdtree.nearest(&query_point, K_NEIGHBORS, &squared_euclidean) {
            Ok(n) => n,
            Err(_) => return vec![0.0, 0.0, 0.0], // エラー時はゼロ法線
        };

        if neighbors.len() < 3 {
            return vec![0.0, 0.0, 0.0];
        }

        // 3. 共分散行列の計算 (メモリ確保なし版)
        // Cov = E[XX^T] - E[X]E[X]^T を利用して1パスで計算する
        // または、重心を求めてから差分を累積する2パスでも、配列確保よりは速い
        
        // --- パス1: 重心 (Centroid) 計算 ---
        let mut sum = Vector3::zeros();
        for &(_, idx) in &neighbors {
            let nx = target_pts[[*idx, 0]];
            let ny = target_pts[[*idx, 1]];
            let nz = target_pts[[*idx, 2]];
            sum += Vector3::new(nx, ny, nz);
        }
        let k_f64 = neighbors.len() as f64;
        let centroid = sum / k_f64;

        // --- パス2: 共分散行列 (Covariance Matrix) 計算 ---
        // nalgebra の Matrix3 を使う (スタック確保なので爆速)
        let mut cov = Matrix3::zeros();
        for &(_, idx) in &neighbors {
            let nx = target_pts[[*idx, 0]];
            let ny = target_pts[[*idx, 1]];
            let nz = target_pts[[*idx, 2]];
            
            let d = Vector3::new(nx, ny, nz) - centroid;
            // 外積 (d * d^T) を加算
            cov += d * d.transpose();
        }
        // 通常は N-1 で割るが、固有ベクトルの向きには影響しないので省略可
        // cov /= k_f64; 

        // 4. 固有値分解 (Symmetric Eigen decomposition)
        // 共分散行列は対称行列なので、SVDより高速な SymmetricEigen を使用
        let eigen = SymmetricEigen::new(cov);
        
        // nalgebraのeigenvaluesはVector3なのでイテレータで回して探す
        let (min_idx, _) = eigen.eigenvalues.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // そのインデックスに対応する固有ベクトルを取得
        let mut normal = eigen.eigenvectors.column(min_idx).into_owned();

        // 法線の正規化（念のため）
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

    // Vec<Vec<f64>> -> Array2<f64> への変換 (コストは軽微)
    let mut normals_arr = Array2::<f64>::zeros((n_points, 3));
    for (i, normal) in normals_vec.into_iter().enumerate() {
        normals_arr[[i, 0]] = normal[0];
        normals_arr[[i, 1]] = normal[1];
        normals_arr[[i, 2]] = normal[2];
    }

    Ok(normals_arr)
}

fn calculate_normals(
    target_pts: &Array2<f64>,
    kdtree: &KdTree<f64, usize, [f64; 3]>,
    viewpoint: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_points = target_pts.nrows();
    let mut normals = Array2::<f64>::zeros((n_points, 3));

    azip!((mut normal_row in normals.axis_iter_mut(Axis(0)),
        p_row in target_pts.axis_iter(Axis(0))) {
        let query_point = [p_row[0], p_row[1], p_row[2]];
        let neighbors = kdtree.nearest(
            &query_point,
            K_NEIGHBORS,
            &squared_euclidean
        ).unwrap();

        let neighbor_indices: Vec<usize> = neighbors.iter()
            .map(|&(_dist, &idx)| idx)
            .collect();

        let neighbor_pts = target_pts.select(Axis(0), &neighbor_indices);

        let centroid = neighbor_pts.mean_axis(Axis(0)).unwrap();
        let centered = &neighbor_pts - &centroid;
        let cov = centered.t().dot(&centered);

        let (_u, _s, vh_opt) = cov.svd(false, true).unwrap();
        let vh = vh_opt.unwrap();
        let mut normal = vh.row(2).to_owned();

        let to_viewpoint = viewpoint - &centroid;
        if normal.dot(&to_viewpoint) < 0.0 {
            normal *= -1.0;
        }

        let norm = normal.mapv(|x| x * x).sum().sqrt();
        if norm < 1e-9 {
            normal.fill(0.0);
        } else {
            normal /= norm;
        }

        normal_row.assign(&normal);
    });

    Ok(normals)
}

fn array2_to_points(
    arr: &Array2<f64>
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
) -> Array2<f64> {
    let n = points.points.len();
    let mut arr = Array2::<f64>::zeros((n, 3));

    for (i, p) in points.points.iter().enumerate() {
        arr[[i, 0]] = p.x as f64;
        arr[[i, 1]] = p.y as f64;
        arr[[i, 2]] = p.z as f64;
    }

    arr
}

fn find_closest_pairs_kdtree(
    source_pts: &Array2<f64>,      // サンプリングされた source 点群
    target_pts: &Array2<f64>,      // target 全体 (インデックスから点を引くため)
    kdtree: &KdTree<f64, usize, [f64; 3]> // 事前に構築した tree
) -> (Array2<f64>, Vec<usize>, Vec<f64>) {
    
    let n = source_pts.nrows();

    let results: Vec<(usize, f64)> = (0..n).into_par_iter()
        .map(|i| {
            let source_row = source_pts.row(i);
            let query_point = [source_row[0], source_row[1], source_row[2]];

            let neighbors = kdtree.nearest(
                &query_point, 
                1, 
                &squared_euclidean
            ).unwrap();

            let (dist_sq, &target_index) = neighbors[0];
            (target_index, dist_sq)
        })
        .collect();

    let (closest_indices, distance_sq): (Vec<usize>, Vec<f64>) = results.into_iter().unzip();
    
    // 見つかったインデックスのリストを使って、
    // target_pts から対応する点を一括で抽出する
    let matched_target_pts = target_pts.select(Axis(0), &closest_indices);
    (matched_target_pts, closest_indices, distance_sq)
}

fn find_closest_pairs(
    source_pts: &Array2<f64>,
    target_pts: &Array2<f64>
) -> (Array2<f64>, Vec<usize>) {
    let n = source_pts.nrows();
    let m = target_pts.nrows();
    let mut dist_matrix = Array2::<f64>::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            let diff = &source_pts.row(i) - &target_pts.row(j);
            dist_matrix[[i, j]] = diff.mapv(|x| x * x).sum().sqrt();
        }
    }
    
    let closest_indices = find_closest_indices(&dist_matrix);
    let matched_target_pts = target_pts.select(Axis(0), &closest_indices);
    (matched_target_pts, closest_indices)
}

fn find_closest_indices(
    dist_matrix: &Array2<f64>
) -> Vec<usize> {
    let n = dist_matrix.nrows();
    let mut closest_indices = Vec::with_capacity(n);

    for i in 0..n {
        let row = dist_matrix.row(i);
        let min_idx = row.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        closest_indices.push(min_idx);
    }
    closest_indices
}

// JSONの構造に合わせた定義
#[derive(Debug, Deserialize)]
struct LivoxImuBatch {
    timestamp: u64, // ナノ秒と仮定 (例: 484350964530)
    angular_velocity: Vec<[f32; 3]>,
    // sample_count は Vecのlen()でわかるので無視してもOK
}

// 扱いやすいように変換した後の1サンプルあたりの構造体
#[derive(Debug, Clone)]
struct ImuSample {
    timestamp_sec: f64, // 計算しやすいように秒単位(f64)に変換して持つ
    gyro: Array1<f64>,
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
                gyro: arr1(&[gyro[0] as f64, gyro[1] as f64, gyro[2] as f64]),
            });
        }
    }

    // 念のため時間順にソート（JSONが順番通りなら不要だが安全のため）
    samples.sort_by(|a, b| a.timestamp_sec.partial_cmp(&b.timestamp_sec).unwrap());

    Ok(samples)
}

/// 指定された開始時刻から duration (秒) の間の平均角速度を計算
fn get_avg_gyro(
    all_samples: &[ImuSample], 
    frame_start_time: f64, 
    duration: f64
) -> Option<Array1<f64>> {
    let frame_end_time = frame_start_time + duration;
    
    let mut sum = Array1::<f64>::zeros(3);
    let mut count = 0;

    // バイナリサーチで開始位置を探すと高速だが、今回は単純なフィルタで実装
    // (データ量が膨大なら skip_while 等で最適化してください)
    for sample in all_samples {
        if sample.timestamp_sec >= frame_start_time && sample.timestamp_sec < frame_end_time {
            sum = sum + &sample.gyro;
            count += 1;
        }
        // 時間を過ぎたらループを抜ける（ソート済み前提）
        if sample.timestamp_sec >= frame_end_time {
             break; // 最適化: これ以上後ろは見なくていい
        }
    }

    if count > 0 {
        Some(sum / (count as f64))
    } else {
        None
    }
}

// --- ヘルパー3: 歪み補正 (Deskewing) ---
fn deskew_point_cloud(
    points: &Array2<f64>,
    angular_velocity: &Array1<f64>,
    scan_duration: f64,
) -> Array2<f64> {
    let n_points = points.nrows();
    let mut corrected_points = Array2::<f64>::zeros((n_points, 3));
    
    // nalgebraのVector3に変換
    let omega = Vector3::new(angular_velocity[0], angular_velocity[1], angular_velocity[2]);

    for i in 0..n_points {
        // 点群が時間順に並んでいる前提で、リニアに時刻を推定
        let ratio = i as f64 / n_points as f64;
        let dt = ratio * scan_duration;

        // 回転ベクトル = 角速度 * 経過時間
        // ※もし補正方向が逆なら -omega * dt にする
        let angle_axis = omega * dt;
        
        // 回転行列
        let rotation = Rotation3::new(angle_axis);
        
        // 座標変換
        let p = Vector3::new(points[[i, 0]], points[[i, 1]], points[[i, 2]]);
        let p_corrected = rotation * p;

        corrected_points[[i, 0]] = p_corrected.x;
        corrected_points[[i, 1]] = p_corrected.y;
        corrected_points[[i, 2]] = p_corrected.z;
    }
    corrected_points
}

/// 距離によるフィルタリング (Pass-through filter based on Range)
/// min_range: これより近い点は削除 (例: 0.5m - 自分自身の映り込み除去)
/// max_range: これより遠い点は削除 (例: 40.0m - 精度低下防止)
fn filter_by_range(points: &Array2<f64>, min_range: f64, max_range: f64) -> Array2<f64> {
    let n_points = points.nrows();
    
    // 結果を格納するバッファ（最大サイズで確保しておくと再確保が起きない）
    let mut valid_indices = Vec::with_capacity(n_points);
    
    let min_sq = min_range * min_range;
    let max_sq = max_range * max_range;

    // 各点の距離判定
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        let z = points[[i, 2]];
        
        // 平方根(sqrt)を取ると重いので、二乗のまま比較するのが高速化のコツ
        let dist_sq = x*x + y*y + z*z;

        if dist_sq >= min_sq && dist_sq <= max_sq {
            valid_indices.push(i);
        }
    }

    // 有効な点だけを抽出して新しいArray2を作る
    points.select(Axis(0), &valid_indices)
}