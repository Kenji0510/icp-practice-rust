use std::time::Instant;

use anyhow::{Context, Result};
use icp_practice::operate_pcd::{PointXYZ, Points, load_pcd_xyz, save_pcd};
use kdtree::{KdTree, distance::squared_euclidean};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
// use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{SVD, Solve};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
// use rayon::prelude::*;

// const TARGET_PCD_PATH: &str = "data/input/H927/lidar-target.pcd";
// const SOURCE_PCD_PATH: &str = "data/input/H927/vggt-source.pcd";
const TARGET_PCD_PATH: &str = "/workspace/input/lidar-target.pcd";
const SOURCE_PCD_PATH: &str = "/workspace/input/vggt-source.pcd";
const OUTPUT_PATH: &str = "data/output";
// const OUTPUT_PATH: &str = "/workspace/output";

const DEFAULT_SAMPLE_SIZE: usize = 7200;
const DEFAULT_TRIM_PERCENTAGE: f64 = 1.0;
const K_NEIGHBORS: usize = 15;
const INIT_ICP_MAX_ITERATIONS: usize = 20;
const SECOND_ICP_MAX_ITERATIONS: usize = 40;
const TOLERANCE: f64 = 0.015;

#[derive(Debug, Serialize, Deserialize)]
struct ICPStatResult {
    label: String,
    save_pcd_path: String,
    rmse: f64,
    transform: Vec<Vec<f64>>,
}

fn main() -> Result<()> {
    let target_pcd_file_path = TARGET_PCD_PATH;
    let source_pcd_file_path = SOURCE_PCD_PATH;

    let target_d = load_pcd_xyz(target_pcd_file_path).context("Failed to load Target PCD")?;
    let source_d = load_pcd_xyz(source_pcd_file_path).context("Failed to load Source PCD")?;

    let start = std::time::Instant::now();

    let target_pts = Points::new(target_d);
    let source_pts = Points::new(source_d);

    let target_pts_arr = points_to_array2(&target_pts);
    let source_pts_arr = points_to_array2(&source_pts);

    println!(
        "Loaded Target: {} pts, Source: {} pts",
        target_pts_arr.nrows(),
        source_pts_arr.nrows()
    );

    // === Centroid Alignment) ===
    let (_, transformed_source_pts_arr) =
        registration_pcd_center(&source_pts_arr, &target_pts_arr);
    println!("Source point cloud centered to target centroid.");

    println!("Building k-d tree & Calculating Normals...");
    let n_dims_target = target_pts_arr.ncols();
    let mut kdtree: KdTree<f64, usize, Vec<f64>> = KdTree::new(n_dims_target);
    for (i, point_row) in target_pts_arr.rows().into_iter().enumerate() {
        kdtree
            .add(point_row.to_slice().unwrap().to_vec(), i)
            .unwrap();
    }

    let viewpoint = arr1(&[0.0, 0.0, 0.0]);
    let target_normals = calculate_normals(&target_pts_arr, &kdtree, &viewpoint)?;
    println!("Preparation complete.");

    // === ICP Point-to-Plane ===
    let initial_transform = Array2::<f64>::eye(4);

    let (final_transform, final_rmse) = perform_icp_point_to_plane(
        &transformed_source_pts_arr,
        &target_pts_arr,
        &target_normals,
        &kdtree,
        initial_transform,
        INIT_ICP_MAX_ITERATIONS,
        TOLERANCE,
        DEFAULT_SAMPLE_SIZE,
        DEFAULT_TRIM_PERCENTAGE,
    )?;

    println!("ICP Result: RMSE = {:.6}", final_rmse);
    println!("Final Transform:\n{:?}", final_transform);

    // --- 2h. 最終結果の計算 ---
    let n_points_source = transformed_source_pts_arr.nrows();
    let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    source_homogeneous
        .slice_mut(s![.., 0..3])
        .assign(&transformed_source_pts_arr);

    let _ = source_homogeneous.dot(&final_transform.t());
    // let final_aligned_source_pts_arr = final_transformed_homogeneous.slice(s![.., 0..3]);

    let n_points_source = transformed_source_pts_arr.nrows();
    let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    source_homogeneous
        .slice_mut(s![.., 0..3])
        .assign(&transformed_source_pts_arr);

    let base_aligned_homo = source_homogeneous.dot(&final_transform.t());
    let base_aligned_pts = base_aligned_homo.slice(s![.., 0..3]).to_owned();

    // --- 5. 反転・回転候補の作成と評価 ---
    // 元のsource点群に対して反転・回転を適用してから、重心合わせ→ICPを実行
    println!("\n--- Starting Hypothesis Verification (from original source) ---");

    // 反転・回転の変換行列を4x4行列として定義
    let flip_rot_transforms = vec![
        ("LR_Flip_Reverse-Y", create_lr_flip_matrix()),
        ("UD_Flip_Reverse-Z", create_ud_flip_matrix()),
        ("Front-Back_Flip_Reverse-X", create_fb_flip_matrix()),
        ("Rot_180_Z", create_rot_180_z_matrix()),
        ("Rot_180_Y", create_rot_180_y_matrix()),
        ("Rot_180_X", create_rot_180_x_matrix()),
        // (
        //     "LR+UD_Flip",
        //     create_lr_flip_matrix().dot(&create_ud_flip_matrix()),
        // ),
        // (
        //     "LR+FB_Flip",
        //     create_lr_flip_matrix().dot(&create_fb_flip_matrix()),
        // ),
        // (
        //     "UD+FB_Flip",
        //     create_ud_flip_matrix().dot(&create_fb_flip_matrix()),
        // ),
        ("Rot_90_X", create_rot_90_x_matrix()),
        ("Rot_-90_X", create_rot_minus_90_x_matrix()),
        ("Rot_90_Y", create_rot_90_y_matrix()),
        ("Rot_-90_Y", create_rot_minus_90_y_matrix()),
        ("Rot_90_Z", create_rot_90_z_matrix()),
        ("Rot_-90_Z", create_rot_minus_90_z_matrix()),
    ];

    // 候補点群を生成（元のsourceに対して反転・回転を適用）
    let mut candidates = vec![(
        "Original",
        base_aligned_pts.clone(),
        final_rmse,
        final_transform.clone(),
    )];

    for (label, flip_rot_matrix) in flip_rot_transforms {
        // 1. 元のsource点群に反転・回転変換を適用
        let mut source_homo = Array2::<f64>::ones((source_pts_arr.nrows(), 4));
        source_homo.slice_mut(s![.., 0..3]).assign(&source_pts_arr);
        let flipped_homo = source_homo.dot(&flip_rot_matrix.t());
        let flipped_pts = flipped_homo.slice(s![.., 0..3]).to_owned();

        // 2. 重心合わせ
        let (_, transformed_flipped_pts) = registration_pcd_center(&flipped_pts, &target_pts_arr);

        // 3. ICPを実行
        let (icp_tf, icp_rmse) = perform_icp_point_to_plane(
            &transformed_flipped_pts,
            &target_pts_arr,
            &target_normals,
            &kdtree,
            Array2::eye(4),
            SECOND_ICP_MAX_ITERATIONS,
            TOLERANCE,
            DEFAULT_SAMPLE_SIZE,
            DEFAULT_TRIM_PERCENTAGE,
        )?;

        // 4. 最終的な点群位置を計算
        let mut h = Array2::<f64>::ones((transformed_flipped_pts.nrows(), 4));
        h.slice_mut(s![.., 0..3]).assign(&transformed_flipped_pts);
        let final_homo = h.dot(&icp_tf.t());
        let final_pts = final_homo.slice(s![.., 0..3]).to_owned();

        // 5. 全体の変換行列を計算（デバッグ用）
        let combined_tf = icp_tf.clone();

        candidates.push((label, final_pts, icp_rmse, combined_tf));
        println!("Hypothesis [{}]: RMSE = {:.6}", label, icp_rmse);
    }

    let elapsed = start.elapsed();

    // ベストな候補を選択
    let mut best_rmse = f64::MAX;
    let mut best_label = String::from("None");
    let mut best_final_pts = Array2::<f64>::zeros((0, 3));
    let mut best_tf = Array2::<f64>::eye(4);

    for (label, pts, rmse, tf) in candidates.clone() {
        // Select the best transform
        if rmse < best_rmse {
            best_rmse = rmse;
            best_label = label.to_string();
            best_final_pts = pts.clone();
            best_tf = tf.clone();
        }
    }

    let mut icp_stat_results: Vec<ICPStatResult> = Vec::new();

    println!("\n=== Saving each transformed point cloud ===");
    for (label, pts, rmse, tf) in candidates {
        // Save each transformed data
        let aligned_source_pts = array2_to_points(&pts.to_owned());
        let colored_aligned_source_pts = aligned_source_pts.transform_colored_points((0, 255, 0)); // Green
        let colored_target_pts = target_pts.transform_colored_points((0, 0, 255)); // Blue
        let colored_source_pts = source_pts.transform_colored_points((255, 0, 0)); // Red

        let mut all_points = colored_target_pts.clone();
        all_points.extend(colored_aligned_source_pts.clone());
        all_points.extend(colored_source_pts.clone());

        let save_path = format!(
            "{}/{}.pcd",
            OUTPUT_PATH,
            format!("icp-aligned-by-{}", label)
        );
        match save_pcd(&all_points, &save_path) {
            Ok(_) => println!("Saved aligned points to {}", save_path),
            Err(e) => eprintln!("Failed to save PCD file: {}", e),
        }

        icp_stat_results.push(ICPStatResult {
            label: label.to_string(),
            save_pcd_path: save_path,
            rmse: rmse,
            transform: tf
                .clone()
                .into_raw_vec()
                .chunks(4)
                .map(|row| row.to_vec())
                .collect(),
        });
    }
    println!("==========================================");

    println!("\n=== Saving ICP statistics results ===");
    // Serialize ICP statistics to JSON
    let icp_stat_json = serde_json::to_string_pretty(&icp_stat_results)?;
    let icp_stat_save_path = format!("{}/{}", OUTPUT_PATH, "icp-all-stat-results.json");
    std::fs::write(&icp_stat_save_path, icp_stat_json)?;
    println!("Saved ICP statistics to {}", icp_stat_save_path);
    println!("==========================================");

    println!("\n=== Best ICP Result ===");
    println!("Best: {} (RMSE: {:.6})", best_label, best_rmse);
    println!("Best Transform:\n{:?}", best_tf);
    println!("Elapsed time: {:.2?}", elapsed);

    icp_stat_results.clear();
    icp_stat_results.push(ICPStatResult {
        label: best_label.clone(),
        save_pcd_path: format!("{}/{}.pcd", OUTPUT_PATH, "icp-aligned_best-result"),
        rmse: best_rmse,
        transform: best_tf
            .clone()
            .into_raw_vec()
            .chunks(4)
            .map(|row| row.to_vec())
            .collect(),
    });
    let best_stat_json = serde_json::to_string_pretty(&icp_stat_results)?;
    let best_stat_save_path = format!("{}/{}", OUTPUT_PATH, "icp-best-stat-result.json");
    std::fs::write(&best_stat_save_path, best_stat_json)?;
    println!("Saved best ICP statistics to {}", best_stat_save_path);
    println!("==========================================");

    let aligned_source_pts = array2_to_points(&best_final_pts.to_owned());
    let colored_target_pts = target_pts.transform_colored_points((0, 0, 255)); // Blue
    let colored_source_pts = source_pts.transform_colored_points((255, 0, 0)); // Red
    let colored_aligned_source_pts = aligned_source_pts.transform_colored_points((0, 255, 0)); // Green
    let colored_transformed_source_pts = array2_to_points(&transformed_source_pts_arr.to_owned())
        .transform_colored_points((255, 255, 0)); // Yellow

    let mut all_points = colored_target_pts.clone();
    all_points.extend(colored_aligned_source_pts.clone());
    all_points.extend(colored_source_pts.clone());

    let all_points_without_centering = all_points.clone();

    all_points.extend(colored_transformed_source_pts.clone());

    // Save each point clouds
    let save_path = format!(
        "{}/{}.pcd",
        OUTPUT_PATH, "icp-aligned_best-result-with-centering"
    );
    match save_pcd(&all_points, &save_path) {
        Ok(_) => println!("Saved aligned points to {}", save_path),
        Err(e) => eprintln!("Failed to save PCD file: {}", e),
    }

    let save_path = format!("{}/{}.pcd", OUTPUT_PATH, "icp-aligned_best-result");
    match save_pcd(&all_points_without_centering, &save_path) {
        Ok(_) => println!("Saved aligned points to {}", save_path),
        Err(e) => eprintln!("Failed to save PCD file: {}", e),
    }

    Ok(())
}

// 反転・回転の変換行列を生成する関数群
fn create_lr_flip_matrix() -> Array2<f64> {
    // Y軸に対する鏡像反転 (左右反転)
    arr2(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_ud_flip_matrix() -> Array2<f64> {
    // Z軸に対する鏡像反転 (上下反転)
    arr2(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_fb_flip_matrix() -> Array2<f64> {
    // X軸に対する鏡像反転 (前後反転)
    arr2(&[
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_180_z_matrix() -> Array2<f64> {
    // Z軸周りの180度回転
    arr2(&[
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_180_y_matrix() -> Array2<f64> {
    // Y軸周りの180度回転
    arr2(&[
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_180_x_matrix() -> Array2<f64> {
    // X軸周りの180度回転
    arr2(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_90_x_matrix() -> Array2<f64> {
    // X軸周りの90度回転 (右手系)
    arr2(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_minus_90_x_matrix() -> Array2<f64> {
    // X軸周りの-90度回転
    arr2(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_90_y_matrix() -> Array2<f64> {
    // Y軸周りの90度回転
    arr2(&[
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_minus_90_y_matrix() -> Array2<f64> {
    // Y軸周りの-90度回転
    arr2(&[
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_90_z_matrix() -> Array2<f64> {
    // Z軸周りの90度回転
    arr2(&[
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn create_rot_minus_90_z_matrix() -> Array2<f64> {
    // Z軸周りの-90度回転
    arr2(&[
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn perform_icp_point_to_plane(
    source_pts: &Array2<f64>,              // ソース点群 (Nx3)
    target_pts: &Array2<f64>,              // ターゲット点群 (Mx3)
    target_normals: &Array2<f64>,          // ターゲット法線 (Mx3)
    kdtree: &KdTree<f64, usize, Vec<f64>>, // ターゲットのKDTree
    initial_transform: Array2<f64>,        // 初期変換行列 (4x4)
    max_iterations: usize,
    tolerance: f64,
    sample_size: usize,
    trim_percentage: f64,
) -> Result<(Array2<f64>, f64)> {
    let start_time = Instant::now();
    let n_points_source = source_pts.nrows();

    let mut total_transform = initial_transform;

    let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    source_homogeneous
        .slice_mut(s![.., 0..3])
        .assign(source_pts);

    let mut rng = thread_rng();
    let source_indices: Vec<usize> = (0..n_points_source).collect();
    let mut last_error = f64::MAX;

    for i in 0..max_iterations {
        let current_transformed_homo = source_homogeneous.dot(&total_transform.t());
        let current_pts_arr = current_transformed_homo.slice(s![.., 0..3]).to_owned();

        let (sampled_source_pts, _) = if n_points_source <= sample_size {
            (current_pts_arr.clone(), source_indices.clone())
        } else {
            let indices = source_indices
                .as_slice()
                .choose_multiple(&mut rng, sample_size)
                .cloned()
                .collect::<Vec<usize>>();
            (current_pts_arr.select(Axis(0), &indices), indices)
        };

        // === Nearest Neighbor ===
        let (matched_target_pts, matched_target_indices, distance_sq) =
            find_closest_pairs_kdtree(&sampled_source_pts, target_pts, kdtree);

        // === Outlier Rejection ===
        let mut dist_with_indices: Vec<(f64, usize)> = distance_sq
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, dist)| (dist, idx))
            .collect();
        // 距離順にソート
        dist_with_indices
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_to_keep = (dist_with_indices.len() as f64 * trim_percentage) as usize;
        let inlier_indices: Vec<usize> = dist_with_indices
            .iter()
            .take(n_to_keep)
            .map(|&(_dist, idx)| idx)
            .collect();

        // Extract only inlier points
        let inlier_source_pts = sampled_source_pts.select(Axis(0), &inlier_indices);
        let inlier_target_pts = matched_target_pts.select(Axis(0), &inlier_indices);

        // Extract corresponding normals
        let inlier_target_global_indices: Vec<usize> = inlier_indices
            .iter()
            .map(|&idx_n| matched_target_indices[idx_n])
            .collect();
        let inlier_target_normals = target_normals.select(Axis(0), &inlier_target_global_indices);

        // 5. 微小変換行列の計算 (Point-to-Plane)
        // ここで計算に失敗しても即エラーにせず、Warningを出してスキップまたはbreakする設計
        let delta_transform = match calculate_transformation_pt_to_plane(
            &inlier_source_pts,
            &inlier_target_pts,
            &inlier_target_normals,
        ) {
            Ok(tf) => tf,
            Err(e) => {
                eprintln!("Warning: Transformation solver failed at iter {}: {}", i, e);
                break;
            }
        };

        // 6. 変換行列の更新 T_new = Delta * T_curr
        total_transform = delta_transform.dot(&total_transform);

        // 7. エラー計算 (RMSE)
        let current_error = calculate_mean_pt_to_plane_error(
            &inlier_source_pts,
            &inlier_target_pts,
            &inlier_target_normals,
            &delta_transform,
        );

        if (i + 1) % 5 == 0 || i == 0 {
            println!(
                "Iter {}: RMSE = {:.6} (Inliers: {})",
                i + 1,
                current_error,
                n_to_keep
            );
        }

        // 8. 収束判定
        if current_error < tolerance {
            println!("Converged at iteration {}", i + 1);
            last_error = current_error;
            break;
        }

        if (last_error - current_error).abs() < 1e-6 {
            // println!("Error stabilized.");
            // break;
        }
        last_error = current_error;
    }

    let elapsed = start_time.elapsed();
    println!("ICP function completed in {:.2?}", elapsed);

    Ok((total_transform, last_error))
}

/// Point-to-Plane の平均二乗誤差 (RMSE) を計算する
fn calculate_mean_pt_to_plane_error(
    source_pts: &Array2<f64>, // 適用 "前" のソース点
    target_pts: &Array2<f64>,
    target_normals: &Array2<f64>,
    delta_transform: &Array2<f64>, // "今から" 適用する微小変換
) -> f64 {
    let n = source_pts.nrows();

    // ソース点を同次座標系 (N, 4) に変換
    let mut source_homogeneous = Array2::<f64>::ones((n, 4));
    source_homogeneous
        .slice_mut(s![.., 0..3])
        .assign(source_pts);

    // 微小変換を適用
    let transformed_homogeneous = source_homogeneous.dot(&delta_transform.t());
    let transformed_pts = transformed_homogeneous.slice(s![.., 0..3]);

    // (target - transformed_source) . normal
    let diff = target_pts - &transformed_pts;

    // 各行どうしの内積 (ドット積) を計算
    let errors = (&diff * target_normals)
        .sum_axis(Axis(1)) // 行ごとに合計 (＝内積)
        .mapv(|val| val * val); // 2乗する

    (errors.sum() / n as f64).sqrt() // 二乗平均平方根 (RMSE)
}

// 4x4 の "微小" 変換行列 (Delta T) を返す
fn calculate_transformation_pt_to_plane(
    inlier_source_pts: &Array2<f64>, // 現在のイテレーションのソース点 (N x 3)
    inlier_target_pts: &Array2<f64>, // 対応するターゲット点 (N x 3)
    inlier_target_normals: &Array2<f64>, // 対応するターゲット法線 (N x 3)
) -> Result<Array2<f64>> {
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

    // x = (A^T A)^-1 * (A^T b)
    let x = at_a
        .solve(&at_b)
        .map_err(|e| anyhow::anyhow!("Linear solve failed: {:?}", e))
        .or_else(|_| -> Result<Array1<f64>> {
            // もし A^T A が特異行列 (解けない) なら、SVDで擬似逆行列を使って解く
            println!("Warning: Falling back to SVD solver for linear system.");
            let (u, s, vt) = at_a
                .svd(true, true)
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
    let theta = (alpha * alpha + beta * beta + gamma * gamma).sqrt();
    let r: Array2<f64>; // 3x3 回転行列 R

    if theta < 1e-9 {
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

        // R = I + sin(θ)K + (1-cos(θ))K^2
        r = array![
            [
                k_x * k_x * v_th + c_th,
                k_x * k_y * v_th - k_z * s_th,
                k_x * k_z * v_th + k_y * s_th
            ],
            [
                k_x * k_y * v_th + k_z * s_th,
                k_y * k_y * v_th + c_th,
                k_y * k_z * v_th - k_x * s_th
            ],
            [
                k_x * k_z * v_th - k_y * s_th,
                k_y * k_z * v_th + k_x * s_th,
                k_z * k_z * v_th + c_th
            ]
        ];
    }

    // 3. 4x4 剛体変換行列を構築
    let delta_t = array![
        [r[[0, 0]], r[[0, 1]], r[[0, 2]], tx],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]], ty],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]], tz],
        [0.0, 0.0, 0.0, 1.0]
    ];

    Ok(delta_t)
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

fn registration_pcd_center(
    source_points: &Array2<f64>,
    target_points: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    // Calculate centroids
    let source_centroid = source_points.mean_axis(Axis(0)).unwrap();
    let target_centroid = target_points.mean_axis(Axis(0)).unwrap();

    // Calculate translation
    let translation = &target_centroid - &source_centroid;

    // Create transformation matrix
    let mut transform = Array2::<f64>::eye(4);
    transform[[0, 3]] = translation[0];
    transform[[1, 3]] = translation[1];
    transform[[2, 3]] = translation[2];

    let n_points = source_points.nrows();
    let mut source_copy = Array2::<f64>::ones((n_points, 4));
    source_copy.slice_mut(s![.., 0..3]).assign(source_points);

    let transformed = source_copy.dot(&transform.t());
    let transformed_source = transformed.slice(s![.., 0..3]).to_owned();
    (transform, transformed_source)
}

fn array2_to_points(arr: &Array2<f64>) -> Points {
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

fn points_to_array2(points: &Points) -> Array2<f64> {
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
    source_pts: &Array2<f64>,              // サンプリングされた source 点群
    target_pts: &Array2<f64>,              // target 全体 (インデックスから点を引くため)
    kdtree: &KdTree<f64, usize, Vec<f64>>, // 事前に構築した tree
) -> (Array2<f64>, Vec<usize>, Vec<f64>) {
    let n = source_pts.nrows();

    let results: Vec<(usize, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let source_row = source_pts.row(i);
            let query_point = source_row.as_slice().unwrap();

            let neighbors = kdtree.nearest(query_point, 1, &squared_euclidean).unwrap();

            let (dist_sq, &target_index) = neighbors[0];
            (target_index, dist_sq)
        })
        .collect();

    let (closest_indices, distance_sq): (Vec<usize>, Vec<f64>) = results.into_iter().unzip();

    let matched_target_pts = target_pts.select(Axis(0), &closest_indices);
    (matched_target_pts, closest_indices, distance_sq)
}
