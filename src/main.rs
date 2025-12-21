use std::{collections::VecDeque, fs::File, io::{BufReader, BufWriter}, num::NonZero, usize};

use anyhow::{Result, Context};
use icp_practice::{file_handler::load_pcd_files, operate_pcd::{PointXYZ, PointXYZT, Points, load_pcd_xyzt}, voxelization::voxel_downsample_array2};
use nalgebra::{Matrix3, Rotation3, SymmetricEigen, Unit, UnitQuaternion, Vector3};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
// use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{Inverse, Norm, SVD, Solve};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
// use rayon::prelude::*;

#[derive(Serialize)]
struct PoseData {
    timestamp: f64,
    position: [f32; 3],      // x, y, z
    rotation_quat: [f32; 4], // w, x, y, z
}

#[derive(Serialize)]
struct TrajectoryOutput {
    icp_trajectory: Vec<PoseData>,
    imu_raw_trajectory: Vec<PoseData>,
}

struct FrameData {
    points: Array2<f32>,
    covariances: Vec<Matrix3<f64>>,
}

// const SAMPLE_SIZE: usize = 1500;
// const TRIM_PERCENTAGE: f64 = 1.0;
// const K_NEIGHBORS: usize = 6;
// const MAX_ITERATIONS: usize = 10;
// const TOLERANCE: f32 = 0.2;  // Prev: 0.015
// const VOXEL_SIZE: f32 = 0.4;  // 0.2
const SAMPLE_SIZE: usize = 1500;
const TRIM_PERCENTAGE: f64 = 1.0;
const K_NEIGHBORS: usize = 20;
const MAX_ITERATIONS: usize = 15;
const TOLERANCE: f32 = 0.05;  // Prev: 0.015
const VOXEL_SIZE: f32 = 0.2;  // 0.2

fn main() -> Result<()> {
    let target_pcd_dir = "data/input/mid360/pcd/mid360-20251205-01";
    let pcd_paths = match load_pcd_files(target_pcd_dir) {
        Ok(paths) => paths,
        Err(e) => {
            eprintln!("Error loading PCD files: {}", e);
            return Err(e);
        }
    };
    println!("Found {} PCD files in {}", pcd_paths.len(), target_pcd_dir);

    println!("Loading IMU JSON...");
    let imu_samples = load_and_flatten_imu_json("data/input/mid360/imu/mid360-imu-20251205-01/imu_data.json")
    // let imu_samples = load_imu_json("data/input/mid360/imu/mid360-imu-20251125-03/imu_data.json")
        .context("Failed to load IMU JSON data")?;
    println!("Loaded {} IMU samples.", imu_samples.len());

    if imu_samples.is_empty() {
        return Err(anyhow::anyhow!("IMU data is empty"));
    }
    
    let base_timestamp = imu_samples[0].timestamp_sec;
    println!("Base timestamp set to: {:.3}", base_timestamp);

    let mut icp_trajectory_log: Vec<PoseData> = Vec::new();

    // let initial_pcd = match load_pcd_xyz(pcd_paths[0].to_str().unwrap()) {
    let initial_pcd = match load_pcd_xyzt(pcd_paths[0].to_str().unwrap()) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading initial PCD file: {}", e);
            return Err(e);
        }
    };

    let initial_pts_arr = point_xyzt_to_array2(&initial_pcd);
    // let mut target_pts_arr = initial_pts_arr.clone();

    let mut current_global_pose = Array2::<f32>::eye(4);
    let mut last_delta_transform = Array2::<f32>::eye(4);
    let mut current_velocity = Vector3::<f64>::new(0.0, 0.0, 0.0);
    let mut last_frame_timestamp = base_timestamp;

    // Local map queue
    // let mut local_map_queue: VecDeque<Array2<f32>> = VecDeque::new();
    let mut local_map_queue: VecDeque<FrameData> = VecDeque::new();
    const LOCAL_MAP_SIZE: usize = 20;

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
    
    // local_map_queue.push_back(initial_pts_arr.clone());
    global_map_accumulator.push(initial_pts_arr.clone());

    icp_trajectory_log.push(extract_pose_from_matrix(base_timestamp, &current_global_pose));
    

    for (i, pcd_path) in pcd_paths.iter().enumerate() {
        println!("PCD File {}: {}", i, pcd_path.display());
        if i == 0 {
            continue;
        }

        // Loading current frame pcd
        // let current_pcd = match load_pcd_xyz(pcd_path.to_str().unwrap()) {
        let mut current_pcd = match load_pcd_xyzt(pcd_path.to_str().unwrap()) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Error loading PCD file {}: {}", pcd_path.display(), e);
                continue;
            }
        };

        if current_pcd.is_empty() { continue; }

        let min_timestamp = current_pcd.iter()
            .map(|p| p.timestamp).fold(f64::INFINITY, f64::min);

        // =================================================================
        // ★追加: タイムスタンプの単位変換 (ナノ秒 -> 秒)
        // =================================================================
        // 最初の点のタイムスタンプをチェック
        // 1e16 (10,000,000,000,000,000) 以上ならナノ秒とみなして変換
        let current_frame_timestamp = if min_timestamp > 1e16 {
             min_timestamp / 1_000_000_000.0 
        } else { 
            min_timestamp 
        };

        if i == 1 { // i=0はスキップされているので実質最初のループ
             last_frame_timestamp = current_frame_timestamp;
        }

        let (predicted_pose, predicted_velocity) = predict_pose_by_imu(
            &current_global_pose,  // 前回の確定位置
            &current_velocity,     // 前回の速度
            last_frame_timestamp,  // 前回の時刻
            current_frame_timestamp, // 今回の時刻
            &imu_samples           // IMUデータ全体
        );

        // ★予測結果の表示を追加
        let pred_tx = predicted_pose[[0, 3]];
        let pred_ty = predicted_pose[[1, 3]];
        let pred_tz = predicted_pose[[2, 3]];

        // 前回からの移動距離
        let delta_x = pred_tx - current_global_pose[[0, 3]];
        let delta_y = pred_ty - current_global_pose[[1, 3]];
        let delta_z = pred_tz - current_global_pose[[2, 3]];
        let predicted_distance = (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z).sqrt();

        // 回転角度の計算
        let pred_mat3 = Matrix3::new(
            predicted_pose[[0, 0]] as f64, predicted_pose[[0, 1]] as f64, predicted_pose[[0, 2]] as f64,
            predicted_pose[[1, 0]] as f64, predicted_pose[[1, 1]] as f64, predicted_pose[[1, 2]] as f64,
            predicted_pose[[2, 0]] as f64, predicted_pose[[2, 1]] as f64, predicted_pose[[2, 2]] as f64,
        );
        let curr_mat3 = Matrix3::new(
            current_global_pose[[0, 0]] as f64, current_global_pose[[0, 1]] as f64, current_global_pose[[0, 2]] as f64,
            current_global_pose[[1, 0]] as f64, current_global_pose[[1, 1]] as f64, current_global_pose[[1, 2]] as f64,
            current_global_pose[[2, 0]] as f64, current_global_pose[[2, 1]] as f64, current_global_pose[[2, 2]] as f64,
        );

        let pred_quat = UnitQuaternion::from_matrix(&pred_mat3);
        let curr_quat = UnitQuaternion::from_matrix(&curr_mat3);
        let delta_quat = curr_quat.inverse() * pred_quat;
        let predicted_angle = delta_quat.angle();

        println!("IMU Prediction:");
        println!("  Position: [{:.4}, {:.4}, {:.4}]", pred_tx, pred_ty, pred_tz);
        println!("  Delta: [{:.4}, {:.4}, {:.4}] (distance: {:.4}m)", 
            delta_x, delta_y, delta_z, predicted_distance);
        println!("  Rotation angle: {:.4} rad ({:.2}°)", 
            predicted_angle, predicted_angle.to_degrees());
        println!("  Velocity: [{:.4}, {:.4}, {:.4}] m/s", 
            predicted_velocity.x, predicted_velocity.y, predicted_velocity.z);

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
        let current_pts_arr = preprocess_point_cloud(
            // &current_pts,
            &current_pcd,
            &trajectory,
            0.05 as f32,
            20.0 as f32,
        );
        let elapsed_preprocess = start_time.elapsed();

        // current_pts_arr = voxel_downsample_array2(&current_pts_arr, VOXEL_SIZE);

        // !--- End of Preprocessing ---!

        // Create KdTree for target points
        let start_time = std::time::Instant::now();
        println!("Building k-d tree for each points...");
        // ★GICP変更点2: Sourceの共分散行列を計算
        let source_points_vec: Vec<[f32; 3]> = current_pts_arr.outer_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();
        let kdtree_source = kiddo::ImmutableKdTree::new_from_slice(&source_points_vec);
        let source_covs = compute_covariances(&current_pts_arr, &kdtree_source);

        // !--- End of KdTree build ---!

        // Downsample source points and covariances together
        let (downsampled_pts, downsampled_covs) = voxel_downsample_with_cov(
            &current_pts_arr, 
            &source_covs, 
            VOXEL_SIZE
        );
        println!("Downsampled source points from {} to {} points.",
            current_pts_arr.nrows(),
            downsampled_pts.nrows()
        );

        // !--- End of Downsampling ---!

        // Combine local map points into target_pts_arr
        let (target_pts_arr, target_covs) = if local_map_queue.is_empty() {
            // 初回フレーム: ダウンサンプル済みの現在フレームを target として使う
            (downsampled_pts.clone(), downsampled_covs.clone())
        } else {
            flatten_local_map(&local_map_queue)?
        };

        // Downsample target points and covariances together
        let (target_pts_arr, target_covs) = voxel_downsample_with_cov(
            &target_pts_arr, 
            &target_covs, 
            VOXEL_SIZE
        );

        let target_points_vec: Vec<[f32; 3]> = target_pts_arr.outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
        let kdtree_target = kiddo::ImmutableKdTree::new_from_slice(&target_points_vec);

        // Concatenate local map points
        // let local_map_views: Vec<_> = local_map_queue.iter()
        //     .map(|p| p.view())
        //     .collect();
        // target_pts_arr = ndarray::concatenate(
        //     Axis(0),
        //     &local_map_views
        // ).context("Failed to concatenate arrays for local map")?;

        // target_pts_arr = voxel_downsample_array2(&target_pts_arr, VOXEL_SIZE);


        // println!("k-d tree built with {} points.", target_pts_arr.nrows());
        let elapsed_kdtree = start_time.elapsed();

        // Sourceの法線を計算 (数千点なので高速)
        let tx = current_global_pose[[0, 3]];
        let ty = current_global_pose[[1, 3]];
        let tz = current_global_pose[[2, 3]];
        
        let viewpoint = arr1(&[tx, ty, tz]); // ローカル座標系での視点
        // let target_normals = calculate_normals_optimized(&target_pts_arr, &kdtree_target, &viewpoint)?;
        // let elapsed_normals = start_time.elapsed() - elapsed_kdtree - elapsed_preprocess;

        // let predicted_pose = last_delta_transform.dot(&current_global_pose);
        // ICPの探索開始位置を予測位置にセット
        let mut total_transform = predicted_pose.clone();

        // Copy source points for current frame
        let source_points_num = current_pts_arr.nrows();
        let mut original_source_pts = Array2::<f32>::ones((source_points_num, 4));
        original_source_pts.slice_mut(s![.., 0..3]).assign(&current_pts_arr);

        let mut rng = thread_rng();
        let source_indices: Vec<usize> = (0..source_points_num).collect();

        let mut prev_fitness_score = f64::MAX;

        let start_icp_time = std::time::Instant::now();
        for i in 0..MAX_ITERATIONS {
            // --- 2b. "現在" のソース点群を計算 ---
            let current_transformed_homogeneous = original_source_pts.dot(&total_transform.t());
            let current_source_pts_arr = current_transformed_homogeneous.slice(s![.., 0..3]).to_owned();

            // let (sampled_source_pts, sampled_source_indices) = 
            //     if source_points_num <= SAMPLE_SIZE {
            //         (current_source_pts_arr.clone(), source_indices.clone())
            //     } else {
            //         let indices = source_indices.as_slice()
            //             .choose_multiple(&mut rng, SAMPLE_SIZE)
            //             .cloned()
            //             .collect::<Vec<usize>>();

            //         (current_source_pts_arr.select(Axis(0), &indices), indices)
            //     };

            let (sampled_source_pts, sampled_source_indices) = (current_source_pts_arr.clone(), source_indices.clone());

            // --- 2c. `find_closest_pairs_kdtree` の呼び出し ---
            let (matched_target_pts, matched_target_indices, distance_sq) = 
                find_closest_pairs_kdtree(&sampled_source_pts, &target_pts_arr, &kdtree_target);

            let mut dist_with_indices: Vec<(f32, usize)> = distance_sq.iter()
                .cloned()
                .enumerate()
                .map(|(idx, dist)| (dist, idx))
                .collect();
            dist_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let n_to_keep = (dist_with_indices.len() as f64 * TRIM_PERCENTAGE) as usize;
            let inlier_local_indices: Vec<usize> = dist_with_indices.iter()
                .take(n_to_keep)
                .map(|&(_dist, idx)| idx)
                .collect();

            // ----------------------------------------------------------------
            // 4. ★追加: 現在のRMSE (Fitness Score) を計算
            // ----------------------------------------------------------------
            // まだ変換行列(delta)を適用する「前」の、現在の位置合わせ状態での誤差
            // GICPの目的はこの誤差（マハラノビス距離に近いが、簡易的にユークリッドRMSEで代用可）を下げること
            let inlier_dists_sum: f32 = inlier_local_indices.iter()
                .map(|&idx| distance_sq[idx]) // distance_sq は既に二乗距離
                .sum();
            
            let current_fitness_score = (inlier_dists_sum as f64 / n_to_keep as f64).sqrt();

            // ソルバーに渡すためのデータを抽出
            let inlier_source_pts = sampled_source_pts.select(Axis(0), &inlier_local_indices);
            let inlier_target_pts = matched_target_pts.select(Axis(0), &inlier_local_indices);

            // ★GICP変更点3: インライアの共分散行列を収集
            // 対応する source_covs と target_covs をインデックスで抽出
            let inlier_source_covs: Vec<Matrix3<f64>> = inlier_local_indices.iter()
                .map(|&local_idx| {
                    let original_idx = sampled_source_indices[local_idx]; 
                    source_covs[original_idx]
                })
                .collect();

            // target_covs: 最近傍点のインデックスが必要
            // -> matched_target_indices[local_idx] でターゲットIDが取れる
            let inlier_target_covs: Vec<Matrix3<f64>> = inlier_local_indices.iter()
                .map(|&local_idx| {
                    let target_idx = matched_target_indices[local_idx];
                    target_covs[target_idx]
                })
                .collect();

            // ★GICP変更点4: ソルバー呼び出し
            let delta_transform = solve_gicp_step(
                &inlier_source_pts,
                &inlier_source_covs,
                &inlier_target_pts,
                &inlier_target_covs,
                &total_transform // 現在の姿勢 (Rの計算に必要)
            )?;

            // --- 2f. "総" 変換行列を更新 ---
            // T_k+1 = DeltaT * T_k
            total_transform = delta_transform.dot(&total_transform);
            icp_trajectory_log.push(extract_pose_from_matrix(min_timestamp, &current_global_pose));

            // ----------------------------------------------------------------
            // 7. ★修正: 収束判定 (Convergence Check)
            // ----------------------------------------------------------------
            let translation_diff = delta_transform.slice(s![0..3, 3]).norm();
            let trace_3x3 = delta_transform[[0, 0]] + delta_transform[[1, 1]] + delta_transform[[2, 2]];
            let rotation_diff = ((trace_3x3 - 1.0) / 2.0).clamp(-1.0, 1.0).acos().abs();

            let error_diff = (prev_fitness_score - current_fitness_score).abs();

            final_errors = current_fitness_score as f32;
            
            // if i > 0 && error_diff < 1e-6 && translation_diff < 1e-4 {
            if final_errors < TOLERANCE {
                println!("Converged at iter {}: RMSE {:.6}", i+1, current_fitness_score);
                break;
            }
            // if translation_diff < 1e-3 && rotation_diff < 1e-4 {
            //     println!("Converged at iteration {}", i + 1);
            //     break;
            // }

            prev_fitness_score = current_fitness_score;
        }
        let elapsed_icp = start_icp_time.elapsed();

        // if final_errors > TOLERANCE * 1.5 {
        //     println!("Skipped GICP update due to high error: {}", final_errors);
        //     continue;
        // }

        println!("Frame {}: Error = {:.4}", i, final_errors);

        // 8. Update State
        let prev_pose = current_global_pose.clone();
        current_global_pose = total_transform.clone();

        let dt = current_frame_timestamp - last_frame_timestamp; 
        // ★修正: 速度フィードバックを切る (安定化のため)
        // GICPが少しでも飛ぶと、それが「猛加速」として次のフレームに悪影響を与えるため、
        // 慣性航法(予測)は「回転のみ」とし、位置の予測速度は0とします。
        current_velocity = Vector3::new(0.0, 0.0, 0.0);
        // どうしても速度予測を使いたい場合は、以下のように上限(クランプ)を設けてください
        if dt > 1e-6 {
            let vel = (current_global_pose.slice(s![0..3, 3]).to_owned() - prev_pose.slice(s![0..3, 3])) / dt as f32;
            // 最大 2.0 m/s に制限
            current_velocity = Vector3::new(vel[0_usize] as f64, vel[1_usize] as f64, vel[2_usize] as f64).cap_magnitude(2.0);
        }

        last_frame_timestamp = current_frame_timestamp;

        // 9. Map Update Logic (マップ更新判定)
        // -----------------------------------------------------------
        // ★修正: 正しい移動量(Delta)の計算
        // Delta = T_current * T_prev^-1
        let pose_delta = current_global_pose.dot(&prev_pose.inv().unwrap());
        
        let t_dist = pose_delta.slice(s![0..3, 3]).norm();
        let tr = pose_delta[[0,0]] + pose_delta[[1,1]] + pose_delta[[2,2]];
        let r_angle = ((tr - 1.0)/2.0).clamp(-1.0, 1.0).acos();

        // 閾値: 10cm移動 OR 3度回転
        const KEYFRAME_DIST: f32 = 0.1; 
        const KEYFRAME_ANGLE: f32 = 0.05; 

        // 初期の10フレームは無条件で更新してマップを育てる
        if i < 10 || t_dist > KEYFRAME_DIST || r_angle > KEYFRAME_ANGLE {
            println!("Update Map! (Frame {}, Dist: {:.3}m, Angle: {:.3}rad)", i, t_dist, r_angle);

            // ローカルマップに追加するのは「現在の推定位置」によってGlobal座標系に変換された点群
            // ダウンサンプル済みの点群を使うと軽量
            let n_pts = downsampled_pts.nrows();
            let mut pts_homo = Array2::<f32>::ones((n_pts, 4));
            pts_homo.slice_mut(s![.., 0..3]).assign(&downsampled_pts);
            
            // 変換: P_global = T_current * P_local
            let pts_global_homo = pts_homo.dot(&current_global_pose.t());
            let pts_global = pts_global_homo.slice(s![.., 0..3]).to_owned();

            // 共分散の回転: C_global = R * C_local * R^T
            let r_mat = current_global_pose.slice(s![0..3, 0..3]);
            let r_nalgebra = Matrix3::new(
                r_mat[[0,0]] as f64, r_mat[[0,1]] as f64, r_mat[[0,2]] as f64,
                r_mat[[1,0]] as f64, r_mat[[1,1]] as f64, r_mat[[1,2]] as f64,
                r_mat[[2,0]] as f64, r_mat[[2,1]] as f64, r_mat[[2,2]] as f64,
            );
            
            let covs_global: Vec<Matrix3<f64>> = downsampled_covs.iter()
                .map(|c| r_nalgebra * c * r_nalgebra.transpose())
                .collect();

            // キューに追加
            local_map_queue.push_back(FrameData {
                points: pts_global.clone(),
                covariances: covs_global
            });
            
            if local_map_queue.len() > LOCAL_MAP_SIZE {
                local_map_queue.pop_front();
            }

            // Global Map Accumulator (保存用) にも追加
            // 毎回保存すると重すぎるので、キーフレーム更新時のタイミングで保存
            // さらにデータ量を減らしたい場合は if i % 5 == 0 {} などで間引く
            if i % 6 == 0 {
                global_map_accumulator.push(pts_global);
            }
        }
        // -----------------------------------------------------------

        // Trajectory Log
        icp_trajectory_log.push(extract_pose_from_matrix(current_frame_timestamp, &current_global_pose));

        // Debug
        if i % 50 == 0 {
            // 最後に global_map_accumulator を全部結合して保存
            let final_map = ndarray::concatenate(Axis(0), &global_map_accumulator.iter().map(|a| a.view()).collect::<Vec<_>>())?;
            let voxelized_final_map = voxel_downsample_array2(&final_map, 0.1);
            let final_map_points = array2_to_points(&voxelized_final_map);
            let debug_save_path = format!("data/output/icp_map/debug/merged_until_{}.pcd", i);
            final_map_points.save_pcd(&debug_save_path, (0, 255, 0))
                .context("Failed to save debug merged PCD file")?;
        }

        println!("Preprocessing time: {:.3?}, k-d tree time: {:.3?}, ICP time: {:.3?}",
            elapsed_preprocess,
            elapsed_kdtree,
            // elapsed_normals,
            elapsed_icp
        );

        total_preprocess_time += elapsed_preprocess;
        total_kdtree_time += elapsed_kdtree;
        // total_normals_time += elapsed_normals;
        total_icp_time += elapsed_icp;
        processed_frame_count += 1;
    }

    println!("Generating raw IMU trajectory (with integration)...");
    
    // 加速度情報を使って位置推定も行う関数を呼び出し
    let imu_trajectory_log = compute_imu_only_trajectory(&imu_samples, base_timestamp);

    let output_data = TrajectoryOutput {
        icp_trajectory: icp_trajectory_log,
        imu_raw_trajectory: imu_trajectory_log,
    };

    let json_path = "data/output/icp_map/myself-position/trajectory_comparison.json";
    let file = File::create(json_path).context("Failed to create JSON output file")?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &output_data).context("Failed to write JSON data")?;
    println!("Saved trajectory data to {}", json_path);

    if processed_frame_count > 0 {
        let avg_preprocess = total_preprocess_time / processed_frame_count;
        let avg_kdtree = total_kdtree_time / processed_frame_count;
        let avg_normals = total_normals_time / processed_frame_count;
        let avg_icp = total_icp_time / processed_frame_count;

        println!("\n--- Average Execution Times (over {} frames) ---", processed_frame_count);
        println!("Avg Preprocessing: {:.3?}", avg_preprocess);
        println!("Avg k-d tree:      {:.3?}", avg_kdtree);
        // println!("Avg Normals:       {:.3?}", avg_normals);
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
    // let final_points = array2_to_points(&target_pts_arr);
    // let final_save_path = "data/output/icp_map/final_merged.pcd";
    let final_map = ndarray::concatenate(Axis(0), &global_map_accumulator.iter().map(|a| a.view()).collect::<Vec<_>>())?;
    let voxelized_final_map = voxel_downsample_array2(&final_map, 0.05);
    let final_map_points = array2_to_points(&voxelized_final_map);
    // final_points.save_pcd(final_save_path, (255, 0, 0))
    //     .context("Failed to save final merged PCD file")?;
    let final_save_path = "data/output/icp_map/final_map/final_merged_map.pcd";
    final_map_points.save_pcd(final_save_path, (255, 0, 0))
        .context("Failed to save final merged PCD file")?;
    println!("Saved final merged point cloud to {}", final_save_path);
    
    Ok(())
}

fn predict_pose_by_imu(
    start_pose_mat: &Array2<f32>, // 前回のGICP収束後の姿勢 (4x4)
    start_vel: &Vector3<f64>,     // 前回の速度
    start_time: f64,              // 前回のタイムスタンプ
    end_time: f64,                // 今回のタイムスタンプ
    imu_samples: &[ImuSample],    // 全IMUデータ
) -> (Array2<f32>, Vector3<f64>) { // (予測姿勢, 予測速度)
    
    // 1. Array2<f32> から nalgebra の型 (Isometry3/UnitQuaternion) に変換
    let tx = start_pose_mat[[0, 3]] as f64;
    let ty = start_pose_mat[[1, 3]] as f64;
    let tz = start_pose_mat[[2, 3]] as f64;
    let mut position = Vector3::new(tx, ty, tz);

    let mat3 = Matrix3::new(
        start_pose_mat[[0, 0]] as f64, start_pose_mat[[0, 1]] as f64, start_pose_mat[[0, 2]] as f64,
        start_pose_mat[[1, 0]] as f64, start_pose_mat[[1, 1]] as f64, start_pose_mat[[1, 2]] as f64,
        start_pose_mat[[2, 0]] as f64, start_pose_mat[[2, 1]] as f64, start_pose_mat[[2, 2]] as f64,
    );
    let mut rotation = UnitQuaternion::from_matrix(&mat3);
    let mut velocity = *start_vel;

    // 重力ベクトル (World frame, Z-upと仮定)
    let gravity = Vector3::new(0.0, 0.0, 9.80665);

    // 2. 指定範囲のIMUデータを抽出
    // 前回の終わりから今回の終わりまでを含めるため少しバッファを持たせるか、厳密にフィルタリングする
    let relevant_samples: Vec<&ImuSample> = imu_samples.iter()
        .filter(|s| s.timestamp_sec > start_time && s.timestamp_sec <= end_time)
        .collect();

    let mut last_t = start_time;

    // 3. 積分 (Dead Reckoning)
    for sample in relevant_samples {
        let dt = sample.timestamp_sec - last_t;
        if dt <= 1e-9 { continue; }

        // --- 回転の更新 (Gyro) ---
        let wx = sample.gyro[0] as f64;
        let wy = sample.gyro[1] as f64;
        let wz = sample.gyro[2] as f64;
        let omega = Vector3::new(wx, wy, wz);
        
        let angle = omega.norm() * dt;
        let axis = if angle < 1e-9 { Vector3::x_axis() } else { Unit::new_normalize(omega) };
        let delta_q = UnitQuaternion::from_axis_angle(&axis, angle);
        
        rotation = rotation * delta_q; // Global frame orientation update
        rotation.renormalize();

        // --- 速度・位置の更新 (Accel) ---
        let ax = sample.linear_acceleration[0] as f64;
        let ay = sample.linear_acceleration[1] as f64;
        let az = sample.linear_acceleration[2] as f64;
        let acc_local = Vector3::new(ax, ay, az);

        // ローカル加速度をグローバルへ変換
        let acc_global = rotation * acc_local;
        
        // 重力除去
        let acc_net = acc_global - gravity;

        // 等加速度運動として積分
        position += velocity * dt + 0.5 * acc_net * dt * dt;
        velocity += acc_net * dt;

        last_t = sample.timestamp_sec;
    }

    // 4. nalgebra -> Array2<f32> (4x4 Matrix) に戻す
    let r_mat = rotation.to_rotation_matrix();
    let r = r_mat.matrix();
    
    let predicted_mat = ndarray::array![
        [r[(0,0)] as f32, r[(0,1)] as f32, r[(0,2)] as f32, position.x as f32],
        [r[(1,0)] as f32, r[(1,1)] as f32, r[(1,2)] as f32, position.y as f32],
        [r[(2,0)] as f32, r[(2,1)] as f32, r[(2,2)] as f32, position.z as f32],
        [0.0,             0.0,             0.0,             1.0]
    ];

    (predicted_mat, velocity)
}

fn flatten_local_map(
    queue: &VecDeque<FrameData>
) -> Result<(Array2<f32>, Vec<Matrix3<f64>>)> {
    
    // 1. 点群 (Array2) の結合
    // ndarray::concatenate は View のリストを受け取るので、各フレームのViewを集めます
    let points_views: Vec<_> = queue.iter()
        .map(|frame| frame.points.view())
        .collect();

    // Axis(0) = 行方向（縦）に結合
    let merged_points = ndarray::concatenate(Axis(0), &points_views)
        .context("Failed to concatenate local map points")?;

    // 2. 共分散 (Vec) の結合
    // 事前にサイズを計算して reserve することで、メモリ確保のオーバーヘッドを防ぎます
    let total_points = merged_points.nrows();
    let mut merged_covs = Vec::with_capacity(total_points);

    for frame in queue {
        // extend_from_slice は高速にコピーを行います (Matrix3はCopy/Clone可能)
        merged_covs.extend_from_slice(&frame.covariances);
    }

    // 整合性チェック (念のため)
    if merged_points.nrows() != merged_covs.len() {
        return Err(anyhow::anyhow!(
            "Mismatch between points count ({}) and covariances count ({}) in local map",
            merged_points.nrows(),
            merged_covs.len()
        ));
    }

    Ok((merged_points, merged_covs))
}

fn voxel_downsample_with_cov(
    pts: &Array2<f32>,
    covs: &[Matrix3<f64>],
    voxel_size: f32
) -> (Array2<f32>, Vec<Matrix3<f64>>) {
    let mut grid = std::collections::HashMap::new();

    for i in 0..pts.nrows() {
        let x = pts[[i, 0]];
        let y = pts[[i, 1]];
        let z = pts[[i, 2]];
        let p_vec = Vector3::new(x, y, z);

        let ix = (x / voxel_size).floor() as i32;
        let iy = (y / voxel_size).floor() as i32;
        let iz = (z / voxel_size).floor() as i32;
        let key = (ix, iy, iz);

        grid.entry(key)
            // 既にボクセルに点がある場合：座標を足し合わせ、カウントを増やす
            .and_modify(|(sum, count, _)| {
                *sum += p_vec;
                *count += 1;
            })
            // 初めての点の場合：座標、カウント1、そして共分散を保存
            .or_insert((p_vec, 1, covs[i]));
    }

    // 抽出（重心を計算）
    let n_kept = grid.len();
    let mut new_pts = Array2::<f32>::zeros((n_kept, 3));
    let mut new_covs = Vec::with_capacity(n_kept);

    // HashMapの順番は不定なので、安定した結果が必要ならキーでソートするなどの工夫が必要ですが、
    // ここでは単純にイテレートします。
    for (k, (_, (sum, count, first_cov))) in grid.iter().enumerate() {
        // 重心 = 合計 / 個数
        let centroid = sum / (*count as f32);
        
        new_pts[[k, 0]] = centroid.x;
        new_pts[[k, 1]] = centroid.y;
        new_pts[[k, 2]] = centroid.z;
        
        // 共分散は「そのボクセルを代表する鋭い分布」として、最初の点のものを採用
        new_covs.push(*first_cov);
    }

    (new_pts, new_covs)
}

/// 4x4行列から PoseData を抽出するヘルパー
fn extract_pose_from_matrix(timestamp: f64, transform: &Array2<f32>) -> PoseData {
    let tx = transform[[0, 3]];
    let ty = transform[[1, 3]];
    let tz = transform[[2, 3]];

    // 回転行列成分を抽出
    let mat3 = Matrix3::new(
        transform[[0, 0]] as f64, transform[[0, 1]] as f64, transform[[0, 2]] as f64,
        transform[[1, 0]] as f64, transform[[1, 1]] as f64, transform[[1, 2]] as f64,
        transform[[2, 0]] as f64, transform[[2, 1]] as f64, transform[[2, 2]] as f64,
    );
    let q = UnitQuaternion::from_matrix(&mat3);

    PoseData {
        timestamp,
        position: [tx, ty, tz],
        rotation_quat: [q.w as f32, q.i as f32, q.j as f32, q.k as f32],
    }
}

fn compute_covariances(
    pts: &Array2<f32>,
    kdtree: &kiddo::ImmutableKdTree<f32, 3>,
) -> Vec<Matrix3<f64>> {
    let n_points = pts.nrows();
    let k_neighbors = NonZero::new(K_NEIGHBORS).unwrap();

    (0..n_points).into_par_iter().map(|i| {
        let qx = pts[[i, 0]];
        let qy = pts[[i, 1]];
        let qz = pts[[i, 2]];
        let query = [qx, qy, qz];

        // 1. 近傍探索
        let neighbors = kdtree.nearest_n::<kiddo::SquaredEuclidean>(&query, k_neighbors);
        
        if neighbors.len() < 5 {
            return Matrix3::identity(); // 点が少なすぎる場合は単位行列（球）
        }

        // 2. 重心 (Mean) 計算
        let mut mean = Vector3::zeros();
        for n in &neighbors {
            let idx = n.item as usize;
            mean += Vector3::new(pts[[idx, 0]] as f64, pts[[idx, 1]] as f64, pts[[idx, 2]] as f64);
        }
        mean /= neighbors.len() as f64;

        // 3. 共分散 (Covariance) 計算
        let mut cov = Matrix3::zeros();
        for n in &neighbors {
            let idx = n.item as usize;
            let p = Vector3::new(pts[[idx, 0]] as f64, pts[[idx, 1]] as f64, pts[[idx, 2]] as f64);
            let d = p - mean;
            cov += d * d.transpose();
        }
        cov /= neighbors.len() as f64;

        // 4. 正則化 (GICP Regularization) - ここが重要！
        // 固有値分解して、分布を「パンケーキ状」に整形する
        let eigen = SymmetricEigen::new(cov);
        let rot = eigen.eigenvectors;
        let mut vals = eigen.eigenvalues;

        let mut pairs: Vec<(f64, usize)> = vals.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let min_idx = pairs[0].1; // 最小固有値のインデックス
        vals[min_idx] = 1e-3;     // 法線方向を薄くする
        vals[pairs[1].1] = 1.0;
        vals[pairs[2].1] = 5.0;

        // let min_idx = pairs[0].1; // 法線（厚み）
        // let mid_idx = pairs[1].1; // 縦方向（高さ）
        // let max_idx = pairs[2].1; // 横方向（幅）

        // vals[min_idx] = 1e-3;     // 法線方向を非常に薄く
        // vals[mid_idx] = 0.7;     // 縦方向を薄く
        // vals[max_idx] = 1.0;      // 横方向はやや薄

        // C = R * S * R^T
        let regularized_cov = rot * Matrix3::from_diagonal(&vals) * rot.transpose();
        
        regularized_cov
    }).collect()
}

/// GICPの1ステップ (線形方程式の構築と求解)
fn solve_gicp_step(
    source_pts: &Array2<f32>,           // 現在位置にあるソース点 (N x 3)
    source_covs: &[Matrix3<f64>],       // ソースの共分散 (初期姿勢での計算値)
    target_pts: &Array2<f32>,           // 対応するターゲット点 (N x 3)
    target_covs: &[Matrix3<f64>],       // 対応するターゲットの共分散
    current_transform: &Array2<f32>,    // 現在の推定変換行列 (4x4)
) -> Result<Array2<f32>> { // 戻り値: 微小移動行列 Delta T

    let n = source_pts.nrows();
    
    // 現在の回転行列 R を抽出 (ソースの共分散を回転させるため)
    let r_curr = Matrix3::new(
        current_transform[[0,0]] as f64, current_transform[[0,1]] as f64, current_transform[[0,2]] as f64,
        current_transform[[1,0]] as f64, current_transform[[1,1]] as f64, current_transform[[1,2]] as f64,
        current_transform[[2,0]] as f64, current_transform[[2,1]] as f64, current_transform[[2,2]] as f64,
    );

    // H = J^T * Omega * J の累積用 (6x6)
    let mut h = Array2::<f64>::zeros((6, 6));
    // b = J^T * Omega * error の累積用 (6x1)
    let mut b = Array1::<f64>::zeros(6);

    // Rayonで並列化して H と b を計算し、最後にsumする
    let (h_sum, b_sum) = (0..n).into_par_iter()
        .map(|i| {
            let p_s = Vector3::new(source_pts[[i,0]] as f64, source_pts[[i,1]] as f64, source_pts[[i,2]] as f64);
            let p_t = Vector3::new(target_pts[[i,0]] as f64, target_pts[[i,1]] as f64, target_pts[[i,2]] as f64);
            
            // 1. マハラノビス距離の重み行列 (Information Matrix) Omega を計算
            // C_sum = C_target + R * C_source * R^T
            let c_s_rot = r_curr * source_covs[i] * r_curr.transpose();
            let c_sum = target_covs[i] + c_s_rot;
            
            // Omega = (C_sum)^-1
            let omega = match c_sum.try_inverse() {
                Some(inv) => inv,
                None => return (Array2::<f64>::zeros((6, 6)), Array1::<f64>::zeros(6)),
            };

            // 2. 誤差ベクトル
            let error = p_t - p_s; // Target - Source

            // 3. ヤコビアン J (6x3 ではなく 3x6 として扱い、J^T * Omega * J を計算)
            // error = p_t - (R * p_s_original + t)
            // 微小回転の線形化: - [p_s]x * w + v
            // J = [Skew(p_s), -I] (※定義により符号は変わるが、ここでは標準的な構成で)
            
            // Jの各列ベクトル
            // J_rot (p_s とのクロス積成分)
            // [ 0,  z, -y]
            // [-z,  0,  x]
            // [ y, -x,  0]
            let x = p_s.x; let y = p_s.y; let z = p_s.z;
            
            // 行列演算のために ndarray 形式に変換しつつ計算
            // J^T * Omega * J を作るのが面倒なので、要素ごとに構築するアプローチ
            
            // J^T * Omega (6 x 3)
            // J_rot^T * Omega
            // J_trans^T * Omega
            
            // ここでは簡易的に、J^T * Omega * error と J^T * Omega * J を計算
            
            // Omega * error (3x1)
            let w_e = omega * error;
            
            let mut local_b = Array1::<f64>::zeros(6);
            
            // Rotational part of b: (p_s x (Omega * error))
            let cross = p_s.cross(&w_e);
            local_b[0] = cross.x;
            local_b[1] = cross.y;
            local_b[2] = cross.z;
            
            // Translational part of b: Omega * error
            local_b[3] = w_e.x;
            local_b[4] = w_e.y;
            local_b[5] = w_e.z;

            // H = J^T * Omega * J の構築
            let mut local_h = Array2::<f64>::zeros((6, 6));
            
            // Omega * J_rot (3x3) = Omega * Skew(p_s)
            //   [ 0,  z, -y]
            // S=[-z,  0,  x]
            //   [ y, -x,  0]
            // Col0 = Omega * [0, -z, y]^T
            let s_col0 = Vector3::new(0.0, -z, y);
            let s_col1 = Vector3::new(z, 0.0, -x);
            let s_col2 = Vector3::new(-y, x, 0.0);
            
            let w_s0 = omega * s_col0;
            let w_s1 = omega * s_col1;
            let w_s2 = omega * s_col2;

            // 左上: J_rot^T * Omega * J_rot
            // (Skew(p_s)^T * [w_s0, w_s1, w_s2])
            // Skew^T = -Skew なので、cross productを使って計算可能
            // col0 = p_s x w_s0
            let h00 = p_s.cross(&w_s0);
            let h01 = p_s.cross(&w_s1);
            let h02 = p_s.cross(&w_s2);
            
            local_h[[0,0]] = h00.x; local_h[[0,1]] = h01.x; local_h[[0,2]] = h02.x;
            local_h[[1,0]] = h00.y; local_h[[1,1]] = h01.y; local_h[[1,2]] = h02.y;
            local_h[[2,0]] = h00.z; local_h[[2,1]] = h01.z; local_h[[2,2]] = h02.z;

            // 右下: J_trans^T * Omega * J_trans = Omega (そのもの)
            local_h[[3,3]] = omega[(0,0)]; local_h[[3,4]] = omega[(0,1)]; local_h[[3,5]] = omega[(0,2)];
            local_h[[4,3]] = omega[(1,0)]; local_h[[4,4]] = omega[(1,1)]; local_h[[4,5]] = omega[(1,2)];
            local_h[[5,3]] = omega[(2,0)]; local_h[[5,4]] = omega[(2,1)]; local_h[[5,5]] = omega[(2,2)];

            // 右上: J_rot^T * Omega * J_trans = Skew(p)^T * Omega
            // 行列としては [w_s0, w_s1, w_s2]^T (転置されているため)
            local_h[[0,3]] = w_s0.x; local_h[[0,4]] = w_s0.y; local_h[[0,5]] = w_s0.z;
            local_h[[1,3]] = w_s1.x; local_h[[1,4]] = w_s1.y; local_h[[1,5]] = w_s1.z;
            local_h[[2,3]] = w_s2.x; local_h[[2,4]] = w_s2.y; local_h[[2,5]] = w_s2.z;

            // 左下: 対称行列なので右上の転置
            local_h[[3,0]] = local_h[[0,3]]; local_h[[3,1]] = local_h[[1,3]]; local_h[[3,2]] = local_h[[2,3]];
            local_h[[4,0]] = local_h[[0,4]]; local_h[[4,1]] = local_h[[1,4]]; local_h[[4,2]] = local_h[[2,4]];
            local_h[[5,0]] = local_h[[0,5]]; local_h[[5,1]] = local_h[[1,5]]; local_h[[5,2]] = local_h[[2,5]];

            (local_h, local_b)
        })
        .reduce(
            || (Array2::<f64>::zeros((6, 6)), Array1::<f64>::zeros(6)),
            |mut a, b| {
                a.0 = a.0 + b.0;
                a.1 = a.1 + b.1;
                a
            }
        );

    // H x = b を解く
    // ここは前のコードと同じ (solve or SVD fallback)
    let delta = solve_linear_system_6x6(h_sum, b_sum)?;
    
    Ok(delta)
}

// ヘルパー: 6x6 線形方程式を解いて Delta Transform (4x4) を返す
// (以前の calculate_transformation_pt_to_plane の後半部分と同じロジック)
fn solve_linear_system_6x6(a: Array2<f64>, b: Array1<f64>) -> Result<Array2<f32>> {
    let x = a.solve(&b).or_else(|_| {
         // SVD Fallback (省略または前回のコードを流用)
         // 簡易的に単位行列を返すかエラーにする
         Err(anyhow::anyhow!("Linear solve failed"))
    })?;

    // x = [alpha, beta, gamma, tx, ty, tz]
    // Rodrigues' formula 等で 4x4 行列化 (前回のコード参照)
    // ここでは省略していますが、必ず前回のロジックで実装してください
    let delta_matrix = convert_se3_to_matrix4(x);
    Ok(delta_matrix)
}

// [alpha, beta, gamma, tx, ty, tz] -> 4x4 matrix
fn convert_se3_to_matrix4(x: Array1<f64>) -> Array2<f32> {
    let alpha = x[0]; let beta = x[1]; let gamma = x[2];
    let tx = x[3]; let ty = x[4]; let tz = x[5];

    let theta = (alpha*alpha + beta*beta + gamma*gamma).sqrt();
    let r: Array2<f64>;

    if theta < 1e-9 {
        r = ndarray::array![
            [1.0, -gamma, beta],
            [gamma, 1.0, -alpha],
            [-beta, alpha, 1.0]
        ];
    } else {
        let k_x = alpha / theta;
        let k_y = beta / theta;
        let k_z = gamma / theta;
        let c = theta.cos();
        let s = theta.sin();
        let v = 1.0 - c;

        r = ndarray::array![
            [k_x*k_x*v + c,     k_x*k_y*v - k_z*s, k_x*k_z*v + k_y*s],
            [k_x*k_y*v + k_z*s, k_y*k_y*v + c,     k_y*k_z*v - k_x*s],
            [k_x*k_z*v - k_y*s, k_y*k_z*v + k_x*s, k_z*k_z*v + c]
        ];
    }

    ndarray::array![
        [r[[0,0]] as f32, r[[0,1]] as f32, r[[0,2]] as f32, tx as f32],
        [r[[1,0]] as f32, r[[1,1]] as f32, r[[1,2]] as f32, ty as f32],
        [r[[2,0]] as f32, r[[2,1]] as f32, r[[2,2]] as f32, tz as f32],
        [0.0, 0.0, 0.0, 1.0]
    ]
}

/// IMUデータのみを使って全期間の軌跡（回転 + 位置）を計算する
/// 加速度の二重積分を行うため、時間が経つにつれて位置ズレ（ドリフト）が激しくなります。
fn compute_imu_only_trajectory(imu_samples: &[ImuSample], start_time: f64) -> Vec<PoseData> {
    let mut trajectory = Vec::new();
    
    // 状態変数
    let mut position = Vector3::new(0.0, 0.0, 0.0);
    let mut velocity = Vector3::new(0.0, 0.0, 0.0);
    let mut rotation = UnitQuaternion::identity();

    // 重力ベクトル (World frame, Z-upと仮定: 9.80665 m/s^2)
    // ※ Livox Mid-360の設置向きによって異なりますが、ここでは標準的なZ軸上向きと仮定します
    // ※ 厳密には最初の静止状態で重力方向を推定する必要がありますが、簡易版として固定値を使います
    let gravity = Vector3::new(0.0, 0.0, 9.80665);

    let mut last_time = start_time;

    // 最初の点を追加
    trajectory.push(PoseData {
        timestamp: start_time,
        position: [position.x as f32, position.y as f32, position.z as f32],
        rotation_quat: [rotation.w as f32, rotation.i as f32, rotation.j as f32, rotation.k as f32],
    });

    for sample in imu_samples {
        if sample.timestamp_sec < start_time {
            continue;
        }

        let dt = sample.timestamp_sec - last_time;
        if dt <= 1e-9 { continue; }

        // 1. ジャイロによる回転の更新
        let wx = sample.gyro[0] as f64;
        let wy = sample.gyro[1] as f64;
        let wz = sample.gyro[2] as f64;
        let omega = Vector3::new(wx, wy, wz);
        
        let angle = omega.norm() * dt;
        let axis = if angle < 1e-9 { Vector3::x_axis() } else { Unit::new_normalize(omega) };
        let delta_q = UnitQuaternion::from_axis_angle(&axis, angle);
        
        rotation = rotation * delta_q;
        rotation.renormalize();

        // 2. 加速度による位置の更新
        let ax = sample.linear_acceleration[0] as f64;
        let ay = sample.linear_acceleration[1] as f64;
        let az = sample.linear_acceleration[2] as f64;
        let acc_local = Vector3::new(ax, ay, az);

        // ローカル座標の加速度をグローバル座標系へ変換
        let acc_global = rotation * acc_local;

        // 重力除去 (Linear acceleration = Measured - Gravity)
        // 加速度センサは「重力と逆方向の力」を計測しているため、
        // 静止時は(0,0,1g)を出力します。そこから(0,0,1g)を引くことで運動加速度を得ます。
        let acc_net = acc_global - gravity;

        // 速度更新 (v = v + a*dt)
        velocity += acc_net * dt;

        // 位置更新 (p = p + v*dt + 0.5*a*dt^2)
        position += velocity * dt + 0.5 * acc_net * dt * dt;

        last_time = sample.timestamp_sec;

        trajectory.push(PoseData {
            timestamp: sample.timestamp_sec,
            position: [position.x as f32, position.y as f32, position.z as f32],
            rotation_quat: [rotation.w as f32, rotation.i as f32, rotation.j as f32, rotation.k as f32],
        });
    }

    trajectory
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

    // 1. このフレームの基準となる回転を取得（通常は先頭の点の時刻）
    // points[0]が最も早い時刻であると仮定
    let frame_start_time = points[0].timestamp; // 必要に応じて time_offset 加算
    let start_rotation = get_rotation_at_time(trajectory, frame_start_time);
    
    // 基準回転の逆行列を事前に計算（R_start^-1）
    let start_rotation_inv = start_rotation.inverse();

    for p in points {
        let x = p.x as f32;
        let y = p.y as f32;
        let z = p.z as f32;

        // 距離フィルタ
        let dist_sq = x * x + y * y + z * z;
        if dist_sq < min_dist * min_dist || dist_sq > max_dist * max_dist {
            continue;
        }

        // 2. その点の時刻の回転を取得 (R_current)
        let point_time = p.timestamp; // 必要に応じて time_offset 加算
        let current_rotation = get_rotation_at_time(trajectory, point_time);

        // 3. 相対回転 (Relative Rotation) を計算
        // R_relative = R_start^-1 * R_current
        // これにより、フレーム先頭時刻からその点までの「差分回転」が得られます
        let relative_rotation = start_rotation_inv * current_rotation;

        // 4. 座標変換
        let p_vec = Vector3::new(x as f64, y as f64, z as f64);
        
        // ★修正: inverse()せよ、ではなく「相対回転」をそのまま適用
        // センサーが回転した分だけ、点を同じ方向に回して戻してあげるイメージ
        let corrected = relative_rotation * p_vec;

        if corrected.x.is_nan() || corrected.y.is_nan() || corrected.z.is_nan() {
            continue;
        }

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
    linear_acceleration: Vec<[f32; 3]>,
    sample_count: usize,
    // sample_count は Vecのlen()でわかるので無視してもOK
}

// 扱いやすいように変換した後の1サンプルあたりの構造体
#[derive(Debug, Clone)]
struct ImuSample {
    timestamp_sec: f64, // 計算しやすいように秒単位(f64)に変換して持つ
    gyro: Array1<f32>,
    linear_acceleration: Array1<f32>,
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
                linear_acceleration: arr1(&[
                    batch.linear_acceleration[i][0] as f32,
                    batch.linear_acceleration[i][1] as f32,
                    batch.linear_acceleration[i][2] as f32
                ]),
                sample_count: 1,
            });
        }
    }

    // 念のため時間順にソート（JSONが順番通りなら不要だが安全のため）
    samples.sort_by(|a, b| a.timestamp_sec.partial_cmp(&b.timestamp_sec).unwrap());

    Ok(samples)
}