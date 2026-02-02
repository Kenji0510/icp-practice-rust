use std::time::Instant;

use anyhow::{Result, Context};
use icp_practice::operate_pcd::{PointXYZ, PointXYZNormal, Points, load_pcd_xyz, load_pcd_xyzrgb, save_pcd, save_pcd_with_normals};
use kdtree::{KdTree, distance::squared_euclidean};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
// use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{SVD, Solve};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
// use rayon::prelude::*;

// const WIDTH: f64 = 10.0;
// const HEIGHT: f64 = 5.0;
// const ROTATION_ANGLE_DEG: f64 = 25.0;
// const TRANSLATION_X: f64 = 5.0;
// const TRANSLATION_Y: f64 = 3.0;
// const NOISE_LEVEL: f64 = 0.1; // ノイズを少し強めに
const DEFAULT_SAMPLE_SIZE: usize = 7200;
const DEFAULT_TRIM_PERCENTAGE: f64 = 1.0; // 必要に応じて調整 (例: 0.90)
const K_NEIGHBORS: usize = 15;

fn main() -> Result<()> {
    // let target_pcd_file_path = "data/input/H927/lab-room_voxel_025_xyz_only.pcd";
    let target_pcd_file_path = "data/input/aist/aist-voxelized-025.pcd";
    let source_pcd_file_path = "data/input/aist/vggt-sansouken-room-scale-7_5_voxel_025_xyz_only.pcd";

    // let target_d = load_pcd_xyz(target_pcd_file_path).context("Failed to load Target PCD")?;
    let target_d = load_pcd_xyz(target_pcd_file_path).context("Failed to load Target PCD")?;
    let source_d = load_pcd_xyz(source_pcd_file_path).context("Failed to load Source PCD")?;

    let target_pts = Points::new(target_d);
    let source_pts = Points::new(source_d);
    
    let target_pts_arr = points_to_array2(&target_pts);
    let source_pts_arr = points_to_array2(&source_pts);

    println!("Loaded Target: {} pts, Source: {} pts", target_pts_arr.nrows(), source_pts_arr.nrows());

    // --- 2. 前処理: 重心合わせ (Centroid Alignment) ---
    // ここで大まかな位置を合わせる（必須）
    let (initial_translation, transformed_source_pts_arr) = registration_pcd_center(
        &source_pts_arr,
        &target_pts_arr
    );
    println!("Source point cloud centered to target centroid.");

    // --- 3. 準備: Target側のKDTreeと法線計算 (1回だけ計算して使い回す) ---
    println!("Building k-d tree & Calculating Normals...");
    let n_dims_target = target_pts_arr.ncols();
    let mut kdtree: KdTree<f64, usize, Vec<f64>> = KdTree::new(n_dims_target);
    for (i, point_row) in target_pts_arr.rows().into_iter().enumerate() {
        kdtree.add(point_row.to_slice().unwrap().to_vec(), i).unwrap();
    }

    let viewpoint = arr1(&[0.0, 0.0, 0.0]);
    let target_normals = calculate_normals(&target_pts_arr, &kdtree, &viewpoint)?;
    println!("Preparation complete.");

    // --- 4. ICPの実行 (関数呼び出し) ---
    // ここで重心合わせ済みの `transformed_source_pts_arr` を入力とする
    // 初期変換行列は Identity (重心合わせ済みのため)
    let initial_transform = Array2::<f64>::eye(4);

    let (final_transform, final_rmse) = perform_icp_point_to_plane(
        &transformed_source_pts_arr, // 重心合わせ済みの点群
        &target_pts_arr,
        &target_normals,
        &kdtree,
        initial_transform,           // 初期姿勢
        20,                          // max_iterations
        0.015,                       // tolerance
        DEFAULT_SAMPLE_SIZE,
        DEFAULT_TRIM_PERCENTAGE
    )?;

    println!("ICP Result: RMSE = {:.6}", final_rmse);
    println!("Final Transform:\n{:?}", final_transform);

    // --- 2h. 最終結果の計算 ---
    let n_points_source = transformed_source_pts_arr.nrows();
    let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    source_homogeneous.slice_mut(s![.., 0..3]).assign(&transformed_source_pts_arr);
    
    let final_transformed_homogeneous = source_homogeneous.dot(&final_transform.t());
    let final_aligned_source_pts_arr = final_transformed_homogeneous.slice(s![.., 0..3]);

    let n_points_source = transformed_source_pts_arr.nrows();
    let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    source_homogeneous.slice_mut(s![.., 0..3]).assign(&transformed_source_pts_arr);
    
    let base_aligned_homo = source_homogeneous.dot(&final_transform.t());
    let base_aligned_pts = base_aligned_homo.slice(s![.., 0..3]).to_owned();

    // 2. 反転・回転の候補を作成
    // candidate_original は「そのまま」のデータ
    let (candidate_lr, candidate_ud, candidate_rot_z, candidate_rot_y, candidate_rot_x) = reverse_pcd(&base_aligned_pts);

    // 比較ループ用のベクターを作成
    // (ラベル, 点群データ, 元のRMSE)
    let candidates = vec![
        ("Original",    base_aligned_pts,  final_rmse), 
        ("LR Flip",     candidate_lr,      f64::MAX),
        ("UD Flip",     candidate_ud,      f64::MAX),
        ("Rot 180 Z",   candidate_rot_z,   f64::MAX),
        ("Rot 180 Y",   candidate_rot_y,   f64::MAX),
        ("Rot 180 X",   candidate_rot_x,   f64::MAX),
    ];

    let mut best_rmse = f64::MAX;
    let mut best_label = String::from("None");
    let mut best_final_pts = Array2::<f64>::zeros((0, 3)); // 最終的な点群保持用
    let mut best_tf = Array2::<f64>::eye(4);

    println!("--- Starting Hypothesis Verification ---");

    for (label, pts, pre_calced_rmse) in candidates {
        let (final_pts, rmse) = if label == "Original" {
            // Originalは既にICP済みなのでそのまま採用
            (pts, pre_calced_rmse)
        } else {
            // 反転・回転させた候補に対して、仕上げのICPを実行
            // 初期姿勢は Identity (既に反転などで移動済みのため)
            let (refine_tf, refine_rmse) = perform_icp_point_to_plane(
                &pts,                // 反転済みの点群を入力
                &target_pts_arr,
                &target_normals,
                &kdtree,
                Array2::eye(4),      // initial_transform
                40,                  // max_iterations (仕上げなので多めでもOK)
                0.015,               // tolerance
                DEFAULT_SAMPLE_SIZE,
                DEFAULT_TRIM_PERCENTAGE
            )?;

            // 仕上げICPの結果を点群に適用
            let mut h = Array2::<f64>::ones((pts.nrows(), 4));
            h.slice_mut(s![.., 0..3]).assign(&pts);
            let refined_h = h.dot(&refine_tf.t());
            let refined_pts = refined_h.slice(s![.., 0..3]).to_owned();

            (refined_pts, refine_rmse)
        };

        println!("Hypothesis [{}]: Final RMSE = {:.6}", label, rmse);

        // ベストスコア更新チェック
        if rmse < best_rmse {
            best_rmse = rmse;
            best_label = label.to_string();
            best_final_pts = final_pts;
            best_tf = final_transform.clone();
        }
    }

    println!("----------------------------------------");
    println!("Best: {} (RMSE: {:.6})", best_label, best_rmse);
    println!("Best Transform:\n{:?}", best_tf);

    // plot_points(&source_pts, &target_pts, &current_source_pts, "icp_final.png", "Final State").unwrap();

    // let aligned_source_pts = array2_to_points(&final_aligned_source_pts_arr.to_owned());
    let aligned_source_pts = array2_to_points(&best_final_pts.to_owned());
    let colored_target_pts = target_pts.transform_colored_points((0, 0, 255)); // 青
    let colored_source_pts = source_pts.transform_colored_points((255, 0, 0)); // 赤
    let colored_aligned_source_pts = aligned_source_pts.transform_colored_points((0, 255, 0)); // 緑
    let colored_transformed_source_pts = array2_to_points(&transformed_source_pts_arr.to_owned())
        .transform_colored_points((255, 255, 0)); // 黄色

    let mut all_points = colored_target_pts.clone();
    all_points.extend(colored_aligned_source_pts.clone());
    all_points.extend(colored_source_pts.clone());

    let all_points_without_centering = all_points.clone();

    all_points.extend(colored_transformed_source_pts.clone());

    // Save each point clouds
    let save_path = "data/output/aist/icp_aligned_all_result_v-025.pcd";
    match save_pcd(&all_points, save_path) {
        Ok(_) => println!("Saved aligned points to {}", save_path),
        Err(e) => eprintln!("Failed to save PCD file: {}", e),
    }

    let save_path = "data/output/aist/icp_aligned_result_v-025.pcd";
    match save_pcd(&all_points_without_centering, save_path) {
        Ok(_) => println!("Saved aligned points to {}", save_path),
        Err(e) => eprintln!("Failed to save PCD file: {}", e),
    }

    Ok(())
}

fn reverse_pcd(
    points: &Array2<f64>
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    // 1. 左右反転 (Y軸反転 - YZ平面に対する鏡像)
    let mut mirror_lr = points.clone();
    mirror_lr.slice_mut(s![.., 1]).mapv_inplace(|y| -y);

    // 2. 上下反転 (Z軸反転 - XY平面に対する鏡像)
    let mut mirror_ud = points.clone();
    mirror_ud.slice_mut(s![.., 2]).mapv_inplace(|z| -z);

    // 3. Z軸周りの180度回転 (X-Y平面上での180度回転)
    let mut rotate_180_z = points.clone();
    rotate_180_z.slice_mut(s![.., 0]).mapv_inplace(|x| -x);
    rotate_180_z.slice_mut(s![.., 1]).mapv_inplace(|y| -y);

    // 4. Y軸周りの180度回転 (前後反転)
    let mut rotate_180_y = points.clone();
    rotate_180_y.slice_mut(s![.., 0]).mapv_inplace(|x| -x);
    rotate_180_y.slice_mut(s![.., 2]).mapv_inplace(|z| -z);

    // 5. X軸周りの180度回転 (天地前後反転)
    let mut rotate_180_x = points.clone();
    rotate_180_x.slice_mut(s![.., 1]).mapv_inplace(|y| -y);
    rotate_180_x.slice_mut(s![.., 2]).mapv_inplace(|z| -z);

    (mirror_lr, mirror_ud, rotate_180_z, rotate_180_y, rotate_180_x)
}

fn perform_icp_point_to_plane(
    source_pts: &Array2<f64>,         // ソース点群 (Nx3)
    target_pts: &Array2<f64>,         // ターゲット点群 (Mx3)
    target_normals: &Array2<f64>,     // ターゲット法線 (Mx3)
    kdtree: &KdTree<f64, usize, Vec<f64>>, // ターゲットのKDTree
    initial_transform: Array2<f64>,   // 初期変換行列 (4x4)
    max_iterations: usize,
    tolerance: f64,
    sample_size: usize,
    trim_percentage: f64,
) -> Result<(Array2<f64>, f64)> {
    
    let start_time = Instant::now();
    let n_points_source = source_pts.nrows();
    
    // 現在の累積変換行列 (初期値でセット)
    let mut total_transform = initial_transform;

    // ソース点群を同次座標系 (N, 4) に変換して保持 (これはループ内で不変)
    let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    source_homogeneous.slice_mut(s![.., 0..3]).assign(source_pts);

    let mut rng = thread_rng();
    let source_indices: Vec<usize> = (0..n_points_source).collect();
    let mut last_error = f64::MAX;

    for i in 0..max_iterations {
        // 1. 現在の変換を適用して、一時的な点群座標を得る
        // P_curr = P_orig * T^T
        let current_transformed_homo = source_homogeneous.dot(&total_transform.t());
        let current_pts_arr = current_transformed_homo.slice(s![.., 0..3]).to_owned();

        // 2. サンプリング
        let (sampled_source_pts, _) = if n_points_source <= sample_size {
            (current_pts_arr.clone(), source_indices.clone())
        } else {
            let indices = source_indices.as_slice()
                .choose_multiple(&mut rng, sample_size)
                .cloned()
                .collect::<Vec<usize>>();
            (current_pts_arr.select(Axis(0), &indices), indices)
        };

        // 3. 対応点探索 (Nearest Neighbor)
        let (matched_target_pts, matched_target_indices, distance_sq) = 
            find_closest_pairs_kdtree(&sampled_source_pts, target_pts, kdtree);

        // 4. トリミング (Outlier Rejection)
        let mut dist_with_indices: Vec<(f64, usize)> = distance_sq.iter()
            .cloned()
            .enumerate()
            .map(|(idx, dist)| (dist, idx))
            .collect();
        // 距離順にソート
        dist_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let n_to_keep = (dist_with_indices.len() as f64 * trim_percentage) as usize;
        let inlier_indices: Vec<usize> = dist_with_indices.iter()
            .take(n_to_keep)
            .map(|&(_dist, idx)| idx)
            .collect();

        // インライアのみ抽出
        let inlier_source_pts = sampled_source_pts.select(Axis(0), &inlier_indices);
        let inlier_target_pts = matched_target_pts.select(Axis(0), &inlier_indices);

        // 対応する法線の抽出
        let inlier_target_global_indices: Vec<usize> = inlier_indices.iter()
            .map(|&idx_n| matched_target_indices[idx_n])
            .collect();
        let inlier_target_normals = target_normals.select(Axis(0), &inlier_target_global_indices);

        // 5. 微小変換行列の計算 (Point-to-Plane)
        // ここで計算に失敗しても即エラーにせず、Warningを出してスキップまたはbreakする設計
        let delta_transform = match calculate_transformation_pt_to_plane(
            &inlier_source_pts,
            &inlier_target_pts,
            &inlier_target_normals
        ) {
            Ok(tf) => tf,
            Err(e) => {
                eprintln!("Warning: Transformation solver failed at iter {}: {}", i, e);
                break; // または continue
            }
        };

        // 6. 変換行列の更新 T_new = Delta * T_curr
        total_transform = delta_transform.dot(&total_transform);

        // 7. エラー計算 (RMSE)
        let current_error = calculate_mean_pt_to_plane_error(
            &inlier_source_pts,
            &inlier_target_pts,
            &inlier_target_normals,
            &delta_transform
        );

        if (i + 1) % 5 == 0 || i == 0 {
             println!("Iter {}: RMSE = {:.6} (Inliers: {})", i + 1, current_error, n_to_keep);
        }

        // 8. 収束判定
        if current_error < tolerance {
            println!("Converged at iteration {}", i + 1);
            last_error = current_error;
            break;
        }
        
        // エラーの変化が極小なら止める判定を入れても良い
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

fn registration_pcd_center(
    source_points: &Array2<f64>,
    target_points: &Array2<f64>
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
