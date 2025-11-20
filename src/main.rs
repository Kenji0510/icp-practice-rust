use std::{fs::File, io::BufReader};

use anyhow::{Result, Context};
use icp_practice::{file_handler::load_pcd_files, operate_pcd::{PointXYZ, PointXYZNormal, Points, load_pcd_xyz, save_pcd, save_pcd_with_normals}, voxelization::voxel_downsample_array2};
use kdtree::{KdTree, distance::squared_euclidean};
use nalgebra::{Rotation3, Vector3};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
// use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{Inverse, SVD, Solve};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;
// use rayon::prelude::*;

// const WIDTH: f64 = 10.0;
// const HEIGHT: f64 = 5.0;
// const ROTATION_ANGLE_DEG: f64 = 25.0;
// const TRANSLATION_X: f64 = 5.0;
// const TRANSLATION_Y: f64 = 3.0;
// const NOISE_LEVEL: f64 = 0.1; // ノイズを少し強めに
const SAMPLE_SIZE: usize = 600;
const TRIM_PERCENTAGE: f64 = 0.75;
const K_NEIGHBORS: usize = 40;
const MAX_ITERATIONS: usize = 25;
const TOLERANCE: f64 = 0.012;  // Prev: 0.015
const VOXEL_SIZE: f64 = 0.05;

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
        let current_pts = Points::new(current_pcd);
        let current_pts_arr = points_to_array2(&current_pts);

        // このフレームの開始時刻 = 基準時刻 + (インデックス * 0.1秒)
        let current_frame_start_time = base_timestamp + (i as f64 * scan_interval);

        // 対応するIMUデータの平均角速度を取得
        let avg_gyro = get_avg_gyro(
            &imu_samples, 
            current_frame_start_time, 
            scan_interval
        );

        // デバッグ表示: ちゃんと値が取れているか確認
        // println!(" - Time: {:.3}s ~ {:.3}s, Gyro: {:.3?}", 
        //     current_frame_start_time, 
        //     current_frame_start_time + scan_interval, 
        //     avg_gyro
        // );

        // 歪み補正 (Deskewing) 実行
        let mut current_pts_arr = match avg_gyro {
            Some(gyro) => {
                deskew_point_cloud(&current_pts_arr, &gyro, scan_interval)
            }
            None => {
                eprintln!("Warning: No IMU data for frame {} time window [{:.3}, {:.3}), using original points",
                    i, current_frame_start_time, current_frame_start_time + scan_interval);
                current_pts_arr  // Use original points if no IMU data
            }
        };

        // 2. ★ここで距離フィルタを実行★
        // 例: 0.5m 以内(自分)と、30m 以遠(ノイズ)をカット
        // 屋内なら 20.0〜30.0m、屋外でも 50.0m 程度で切るのが一般的
        let current_pts_arr = filter_by_range(&current_pts_arr, 0.1, 20.0);

        // Create KdTree for target points
        println!("Building k-d tree for target points...");
        let n_dims_target = target_pts_arr.ncols();
        let mut kdtree: KdTree<f64, usize, Vec<f64>> = KdTree::new(n_dims_target);

        for (i, point_row) in target_pts_arr.rows().into_iter().enumerate() {
            let point_slice = point_row.as_slice().unwrap();
            kdtree.add(point_slice.to_vec(), i).unwrap();
        }
        println!("k-d tree built with {} points.", target_pts_arr.nrows());

        // Calculate normals for target points
        let viewpoint: Array1<f64> = arr1(&[0.0, 0.0, 0.0]);
        println!("Calculating normals for target points (k={})...", K_NEIGHBORS);
        // let start_normals = std::time::Instant::now();

        let target_normals = calculate_normals(&target_pts_arr, &kdtree, &viewpoint)
            .context("Failed to calculate normals")?;

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
                find_closest_pairs_kdtree(&sampled_source_pts, &target_pts_arr, &kdtree);

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

        let new_delta = total_transform.dot(&current_global_pose.inv().unwrap());

        last_delta_transform = new_delta;
        current_global_pose = total_transform.clone();

        // let cloned_source_pts = original_source_pts.clone();
        // let final_transformed_homogeneous = cloned_source_pts.dot(&total_transform.t());
        // let final_aligned_source_pts_arr = final_transformed_homogeneous.slice(s![.., 0..3]);

        // target_pts_arr = ndarray::concatenate(
        //     Axis(0), 
        //     &[target_pts_arr.view(), final_aligned_source_pts_arr.view()])
        //     .context("Failed to concatenate arrays")?;

        // target_pts_arr = voxel_downsample_array2(&target_pts_arr, VOXEL_SIZE);
        // println!("After voxel downsampling: {} points", target_pts_arr.nrows());

        if i % 10 == 0 {
            println!("Updating map at frame {}", i);
            
            // ソース点群を現在の推定位置に変換
            let cloned_source_pts = original_source_pts.clone();
            let final_transformed_homogeneous = cloned_source_pts.dot(&total_transform.t());
            let final_aligned_source_pts_arr = final_transformed_homogeneous.slice(s![.., 0..3]);

            // 地図に結合
            target_pts_arr = ndarray::concatenate(
                Axis(0), 
                &[target_pts_arr.view(), final_aligned_source_pts_arr.view()])
                .context("Failed to concatenate arrays")?;

            // ボクセルダウンサンプリング（地図が肥大化しないように）
            target_pts_arr = voxel_downsample_array2(&target_pts_arr, VOXEL_SIZE);
        }

        // Debug
        if i % 10 == 0 {
            let merged_points = array2_to_points(&target_pts_arr);
            let debug_save_path = format!("data/output/icp_map/debug/merged_until_{}.pcd", i);
            merged_points.save_pcd(&debug_save_path, (0, 255, 0))
                .context("Failed to save debug merged PCD file")?;
        }
    }

    // Save final merged point cloud
    let final_points = array2_to_points(&target_pts_arr);
    let final_save_path = "data/output/icp_map/final_merged.pcd";
    final_points.save_pcd(final_save_path, (255, 0, 0))
        .context("Failed to save final merged PCD file")?;

    // let target_pcd_file_path = "data/input/avia/voxelized-025_frame_400.pcd";
    // let source_pcd_file_path = "data/input/avia/voxelized-025_frame_410.pcd";
    // let target_d = load_pcd_xyz(target_pcd_file_path)
    //     .context("Failed to load PCD file")?;
    // let source_d = load_pcd_xyz(source_pcd_file_path)
    //     .context("Failed to load PCD file")?;

    // let target_pts = Points::new(target_d);
    // println!("Loaded {} points from {}", target_pts.points.len(), target_pcd_file_path);

    // let source_pts = Points::new(source_d);
    // println!("Loaded {} points from {}", source_pts.points.len(), source_pcd_file_path);

    // let target_pts_arr = points_to_array2(&target_pts);
    // let source_pts_arr = points_to_array2(&source_pts);

    // let max_iterations = 20;
    // let tolerance = 0.015;  // Prev: 1e-2

    // // let current_source_pts_arr = source_pts_arr.clone();
    // // let rng = thread_rng();
    // // let n_points_source = current_source_pts_arr.nrows();
    // // let source_indices: Vec<usize> = (0..n_points_source).collect();

    // println!("Building k-d tree for target points...");
    // let n_dims_target = target_pts_arr.ncols();
    // let mut kdtree: KdTree<f64, usize, Vec<f64>> = KdTree::new(n_dims_target);

    // for (i, point_row) in target_pts_arr.rows().into_iter().enumerate() {
    //     let point_slice = point_row.as_slice().unwrap();
    //     kdtree.add(point_slice.to_vec(), i).unwrap();
    // }
    // println!("k-d tree built with {} points.", target_pts_arr.nrows());

    // let viewpoint: Array1<f64> = arr1(&[0.0, 0.0, 0.0]);
    // println!("Calculating normals for target points (k={})...", K_NEIGHBORS);
    // let start_normals = std::time::Instant::now();

    // let target_normals = calculate_normals(&target_pts_arr, &kdtree, &viewpoint)
    //     .context("Failed to calculate normals")?;
    // let elapsed_normals = start_normals.elapsed();
    // println!("Normals calculated in {:.2?}", elapsed_normals);

    // let target_pts_with_normals = create_points_with_normals(&target_pts_arr, &target_normals);

    // let normals_save_path = "data/output/with-normals/target_with_normals.pcd";
    // match save_pcd_with_normals(&target_pts_with_normals, normals_save_path) {
    //     Ok(_) => println!("Saved target points with normals to {}", normals_save_path),
    //     Err(e) => eprintln!("Failed to save PCD file with normals: {}", e),
    // }

    // let mut total_transform = Array2::<f64>::eye(4);

    // let n_points_source = source_pts_arr.nrows();
    // let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    // source_homogeneous.slice_mut(s![.., 0..3]).assign(&source_pts_arr);

    // let mut rng = thread_rng();
    // let source_indices: Vec<usize> = (0..n_points_source).collect();

    // let start = std::time::Instant::now();
    // for i in 0..max_iterations {
    //     // --- 2b. "現在" のソース点群を計算 ---
    //     // (N, 4) = (N, 4) .dot (4, 4)
    //     let current_transformed_homogeneous = source_homogeneous.dot(&total_transform.t());
    //     // (N, 3) の座標に戻す
    //     let current_source_pts_arr = current_transformed_homogeneous.slice(s![.., 0..3]).to_owned();

    //     // (サンプリングは current_source_pts_arr から行う - 変更なし)
    //     let (sampled_source_pts, _) = 
    //         if n_points_source <= SAMPLE_SIZE {
    //             (current_source_pts_arr.clone(), source_indices.clone())
    //         } else {
    //             let indices = source_indices.as_slice()
    //                 .choose_multiple(&mut rng, SAMPLE_SIZE)
    //                 .cloned()
    //                 .collect::<Vec<usize>>();

    //             (current_source_pts_arr.select(Axis(0), &indices), indices)
    //         };

    //     // --- 2c. `find_closest_pairs_kdtree` の呼び出し (戻り値が3つに) ---
    //     let (matched_target_pts, matched_target_indices, distance_sq) = 
    //         find_closest_pairs_kdtree(&sampled_source_pts, &target_pts_arr, &kdtree);

    //     // (Trimming (インライア選択) - 変更なし)
    //     let mut dist_with_indices: Vec<(f64, usize)> = distance_sq.iter()
    //         .cloned()
    //         .enumerate()
    //         .map(|(idx, dist)| (dist, idx))
    //         .collect();
    //     dist_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    //     let n_to_keep = (dist_with_indices.len() as f64 * TRIM_PERCENTAGE) as usize;
    //     let inlier_indices: Vec<usize> = dist_with_indices.iter()
    //         .take(n_to_keep)
    //         .map(|&(_dist, idx)| idx)
    //         .collect();
        
    //     // (インライアの点群を取得 - 変更なし)
    //     let inlier_source_pts = sampled_source_pts.select(Axis(0), &inlier_indices);
    //     let inlier_target_pts = matched_target_pts.select(Axis(0), &inlier_indices);

    //     // --- 2d. インライアの "法線" を取得 (★重要★) ---
    //     // `inlier_indices` を使って `matched_target_indices` から "グローバルインデックス" を取得
    //     let inlier_target_global_indices: Vec<usize> = inlier_indices.iter()
    //         .map(|&idx_n| matched_target_indices[idx_n]) // idx_n は 0..N_sample のインデックス
    //         .collect();
    //     // グローバルインデックスを使って `target_normals` から法線を抽出
    //     let inlier_target_normals = target_normals.select(Axis(0), &inlier_target_global_indices);

    //     // --- 2e. "Point-to-Plane" の計算を呼び出し ---
    //     let delta_transform = match calculate_transformation_pt_to_plane(
    //         &inlier_source_pts,
    //         &inlier_target_pts,
    //         &inlier_target_normals
    //     ) {
    //         Ok(tf) => tf,
    //         Err(e) => {
    //             eprintln!("Warning: Failed to solve transformation, skipping iteration: {}", e);
    //             continue; // このイテレーションをスキップ
    //         }
    //     };

    //     // --- 2f. "総" 変換行列を更新 ---
    //     // T_k+1 = DeltaT * T_k
    //     total_transform = delta_transform.dot(&total_transform);
        
    //     // --- 2g. エラー計算 (Point-to-Plane 誤差を推奨) ---
    //     let current_error = calculate_mean_pt_to_plane_error(
    //         &inlier_source_pts, 
    //         &inlier_target_pts, 
    //         &inlier_target_normals,
    //         &delta_transform // "今から" 適用する変換
    //     );
        
    //     println!("Iteration {}: mean pt-to-plane error (from {} inliers, {:.0}% kept) = {}", 
    //         i + 1, 
    //         inlier_indices.len(), 
    //         TRIM_PERCENTAGE * 100.0,
    //         current_error
    //     );

    //     if current_error < tolerance {
    //         println!("Converged at iteration {}", i + 1);
    //         break;
    //     }

    // }
    // let elapsed = start.elapsed();
    // println!("ICP completed in {:.2?}", elapsed);

    // // --- 2h. 最終結果の計算 ---
    // // 最終的な `total_transform` を "元" の `source_homogeneous` に適用
    // let final_transformed_homogeneous = source_homogeneous.dot(&total_transform.t());
    // let final_aligned_source_pts_arr = final_transformed_homogeneous.slice(s![.., 0..3]);

    // println!("Final aligned source points:\n{:?}", total_transform);

    // // plot_points(&source_pts, &target_pts, &current_source_pts, "icp_final.png", "Final State").unwrap();

    // let aligned_source_pts = array2_to_points(&final_aligned_source_pts_arr.to_owned());
    // let colored_target_pts = target_pts.transform_colored_points((0, 0, 255)); // 青
    // let colored_source_pts = source_pts.transform_colored_points((255, 0, 0)); // 赤
    // let colored_aligned_source_pts = aligned_source_pts.transform_colored_points((0, 255, 0)); // 緑

    // let mut all_points = colored_target_pts.clone();
    // all_points.extend(colored_aligned_source_pts.clone());
    // all_points.extend(colored_source_pts.clone());

    // // Save each point clouds
    // let save_path = "data/output/icp_p-to-plane_aligned_result_v-025.pcd";
    // match save_pcd(&all_points, save_path) {
    //     Ok(_) => println!("Saved aligned points to {}", save_path),
    //     Err(e) => eprintln!("Failed to save PCD file: {}", e),
    // }

    Ok(())
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

// (古い `calculate_mean_error` は削除してもOKです)

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

fn calculate_normals(
    target_pts: &Array2<f64>,
    kdtree: &KdTree<f64, usize, Vec<f64>>,
    viewpoint: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_points = target_pts.nrows();
    let mut normals = Array2::<f64>::zeros((n_points, 3));

    azip!((mut normal_row in normals.axis_iter_mut(Axis(0)),
        p_row in target_pts.axis_iter(Axis(0))) {
        let query_point = p_row.to_slice().unwrap();
        let neighbors = kdtree.nearest(
            query_point,
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
    kdtree: &KdTree<f64, usize, Vec<f64>> // 事前に構築した tree
) -> (Array2<f64>, Vec<usize>, Vec<f64>) {
    
    let n = source_pts.nrows();

    let results: Vec<(usize, f64)> = (0..n).into_par_iter()
        .map(|i| {
            let source_row = source_pts.row(i);
            let query_point = source_row.as_slice().unwrap();

            let neighbors = kdtree.nearest(
                query_point, 
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