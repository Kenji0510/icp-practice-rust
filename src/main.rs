use anyhow::{Result, Context};
use icp_practice::operate_pcd::{PointXYZ, PointXYZNormal, Points, load_pcd_xyz, save_pcd, save_pcd_with_normals};
use kdtree::{KdTree, distance::squared_euclidean};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{Determinant, SVD, Solve};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;

// const WIDTH: f64 = 10.0;
// const HEIGHT: f64 = 5.0;
// const ROTATION_ANGLE_DEG: f64 = 25.0;
// const TRANSLATION_X: f64 = 5.0;
// const TRANSLATION_Y: f64 = 3.0;
// const NOISE_LEVEL: f64 = 0.1; // ノイズを少し強めに
const SAMPLE_SIZE: usize = 300;
const TRIM_PERCENTAGE: f64 = 0.9;

const K_NEIGHBORS: usize = 15;

fn main() -> Result<()> {
    let target_pcd_file_path = "data/input/avia/voxelized-025_frame_3.pcd";
    let source_pcd_file_path = "data/input/avia/voxelized-025_frame_13.pcd";
    let target_d = load_pcd_xyz(target_pcd_file_path)
        .context("Failed to load PCD file")?;
    let source_d = load_pcd_xyz(source_pcd_file_path)
        .context("Failed to load PCD file")?;

    let target_pts = Points::new(target_d);
    println!("Loaded {} points from {}", target_pts.points.len(), target_pcd_file_path);

    let mut source_pts = Points::new(source_d);
    println!("Loaded {} points from {}", source_pts.points.len(), source_pcd_file_path);

    // OK
    // let transform_matrix = array![
    //     [0.959326, 0.282294, -0.002065, 2.249126],
    //     [-0.282291, 0.959327, 0.001695, 0.171887],
    //     [0.002459, -0.001043, 0.999996, -0.001295],
    //     [0.000000, 0.000000, 0.000000, 1.000000],
    // ];
    // OK
    // let transform_matrix = array![
    //     [0.962047, 0.259369, -0.084812, 2.365178],
    //     [-0.272116, 0.888537, -0.369399, 0.685336],
    //     [-0.020452, 0.378457, 0.925393, 0.015746],
    //     [0.000000, 0.000000, 0.000000, 1.000000],
    // ];
    // OK
    // let transform_matrix = array![
    //     [0.790851, 0.580114, 0.194992, 9.089207],
    //     [-0.467640, 0.778335, -0.418935, -0.078562],
    //     [-0.394800, 0.240130, 0.886832, -3.941626],
    //     [0.000000, 0.000000, 0.000000, 1.000000],
    // ];
    // NG (Iteration 20, 100)
    // let transform_matrix = array![
    //     [-0.613723, 0.085265, 0.784904, 14.261620],
    //     [-0.788805, -0.023871, -0.614180, -1.739003],
    //     [-0.033632, -0.996072, 0.081908, -6.542201],
    //     [0.000000, 0.000000, 0.000000, 1.000000],
    // ];
    // NG
    // let transform_matrix = array![
    // [0.279534, -0.499074, 0.820236, -11.477912],
    // [-0.280491, 0.774577, 0.566883, -16.308172],
    // [-0.918252, -0.388531, 0.076535, -0.585923],
    // [0.000000, 0.000000, 0.000000, 1.000000]
    // ];
    // source_pts.apply_transform(&transform_matrix);

    let target_pts_arr = points_to_array2(&target_pts);
    let source_pts_arr = points_to_array2(&source_pts);

    let max_iterations = 40;
    let tolerance = 0.015;  // Prev: 1e-2

    let mut current_source_pts_arr = source_pts_arr.clone();
    let mut rng = thread_rng();
    let n_points_source = current_source_pts_arr.nrows();
    let source_indices: Vec<usize> = (0..n_points_source).collect();

    println!("Building k-d tree for target points...");
    let n_dims_target = target_pts_arr.ncols();
    let mut kdtree: KdTree<f64, usize, Vec<f64>> = KdTree::new(n_dims_target);

    for (i, point_row) in target_pts_arr.rows().into_iter().enumerate() {
        let point_slice = point_row.as_slice().unwrap();
        kdtree.add(point_slice.to_vec(), i).unwrap();
    }
    println!("k-d tree built with {} points.", target_pts_arr.nrows());

    let viewpoint: Array1<f64> = arr1(&[0.0, 0.0, 0.0]);
    println!("Calculating normals for target points (k={})...", K_NEIGHBORS);
    let start_normals = std::time::Instant::now();

    let target_normals = calculate_normals(&target_pts_arr, &kdtree, &viewpoint)
        .context("Failed to calculate normals")?;
    let elapsed_normals = start_normals.elapsed();
    println!("Normals calculated in {:.2?}", elapsed_normals);

    let target_pts_with_normals = create_points_with_normals(&target_pts_arr, &target_normals);

    let normals_save_path = "data/output/with-normals/target_with_normals.pcd";
    match save_pcd_with_normals(&target_pts_with_normals, normals_save_path) {
        Ok(_) => println!("Saved target points with normals to {}", normals_save_path),
        Err(e) => eprintln!("Failed to save PCD file with normals: {}", e),
    }

    let mut total_transform = Array2::<f64>::eye(4);

    let n_points_source = source_pts_arr.nrows();
    let mut source_homogeneous = Array2::<f64>::ones((n_points_source, 4));
    source_homogeneous.slice_mut(s![.., 0..3]).assign(&source_pts_arr);

    let mut rng = thread_rng();
    let source_indices: Vec<usize> = (0..n_points_source).collect();

    let start = std::time::Instant::now();
    for i in 0..max_iterations {
        // --- 2b. "現在" のソース点群を計算 ---
        // (N, 4) = (N, 4) .dot (4, 4)
        let current_transformed_homogeneous = source_homogeneous.dot(&total_transform.t());
        // (N, 3) の座標に戻す
        let current_source_pts_arr = current_transformed_homogeneous.slice(s![.., 0..3]).to_owned();

        // (サンプリングは current_source_pts_arr から行う - 変更なし)
        let (sampled_source_pts, sampled_indices) = 
            if n_points_source <= SAMPLE_SIZE {
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
        
        println!("Iteration {}: mean pt-to-plane error (from {} inliers, {:.0}% kept) = {}", 
            i + 1, 
            inlier_indices.len(), 
            TRIM_PERCENTAGE * 100.0,
            current_error
        );

        if current_error < tolerance {
            println!("Converged at iteration {}", i + 1);
            break;
        }

    }
    let elapsed = start.elapsed();
    println!("ICP completed in {:.2?}", elapsed);

    // --- 2h. 最終結果の計算 ---
    // 最終的な `total_transform` を "元" の `source_homogeneous` に適用
    let final_transformed_homogeneous = source_homogeneous.dot(&total_transform.t());
    let final_aligned_source_pts_arr = final_transformed_homogeneous.slice(s![.., 0..3]);

    println!("Final aligned source points:\n{:?}", total_transform);

    // plot_points(&source_pts, &target_pts, &current_source_pts, "icp_final.png", "Final State").unwrap();

    let aligned_source_pts = array2_to_points(&final_aligned_source_pts_arr.to_owned());
    let colored_target_pts = target_pts.transform_colored_points((0, 0, 255)); // 青
    let colored_source_pts = source_pts.transform_colored_points((255, 0, 0)); // 赤
    let colored_aligned_source_pts = aligned_source_pts.transform_colored_points((0, 255, 0)); // 緑

    let mut all_points = colored_target_pts.clone();
    all_points.extend(colored_aligned_source_pts.clone());
    all_points.extend(colored_source_pts.clone());

    // Save each point clouds
    let save_path = "data/output/icp_p-to-plane_aligned_result_v-025.pcd";
    match save_pcd(&all_points, save_path) {
        Ok(_) => println!("Saved aligned points to {}", save_path),
        Err(e) => eprintln!("Failed to save PCD file: {}", e),
    }

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
        mut b_val in &mut b,
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
    // let mut closest_indices = Vec::with_capacity(n);
    // let mut distance_sq = Vec::with_capacity(n);

    // source の各点（サンプリングされた点）についてループ
    // for i in 0..n { // <-- O(N_sample)
    //     let source_row = source_pts.row(i);
    //     let query_point = source_row.as_slice().unwrap();

    //     // k-d tree を使って、最も近い点「1個」を探索 (k=1)
    //     // これが O(M) から O(log M) への高速化！
    //     let neighbors = kdtree.nearest(
    //         query_point,
    //         1, // k=1: 最も近い点 1 個だけを探す
    //         &squared_euclidean // 距離計算の方法
    //     ).unwrap();

    //     // kdtree.nearest は [(距離, &インデックス)] のリストを返す
    //     let (dist_sq, &target_index) = neighbors[0];
        
    //     closest_indices.push(target_index);
    //     distance_sq.push(dist_sq);
    // }

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

fn calculate_transformation(
    source_pts: &Array2<f64>,
    target_pts: &Array2<f64>
) -> (Array2<f64>, Array1<f64>) {
    let centroid_source = source_pts.mean_axis(Axis(0)).unwrap();
    let centroid_target = target_pts.mean_axis(Axis(0)).unwrap();

    let source_prime = source_pts - &centroid_source;
    let target_prime = target_pts - &centroid_target;

    let W = target_prime.t().dot(&source_prime);
    
    // SVD
    let (u, _s, vh) = W.svd(true, true).unwrap();
    let u = u.unwrap();
    let mut vh = vh.unwrap();

    let mut R = u.dot(&vh);
    if R.det().unwrap() < 0.0 {
        let n = vh.nrows();
        for j in 0..vh.ncols() {
            vh[[n - 1, j]] *= -1.0;
        }
        R = u.dot(&vh);
    }

    let t = &centroid_target - &R.dot(&centroid_source);
    
    (R, t)
}

fn calculate_mean_error(
    source_pts: &Array2<f64>,
    target_pts: &Array2<f64>,
) -> f64 {
    let diff = source_pts - target_pts;
    let distances = diff.mapv(|x| x * x)
        .sum_axis(Axis(1))
        .mapv(|x| x.sqrt());
    distances.mean().unwrap()
}

fn plot_points(
    original_source: &Array2<f64>,
    target: &Array2<f64>,
    current_source: &Array2<f64>,
    filename: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-2f64..15f64, -2f64..10f64)?;

    chart.configure_mesh().draw()?;

    // Target points (青)
    chart.draw_series(
        target.rows().into_iter().map(|row| {
            Circle::new((row[0], row[1]), 5, BLUE.filled())
        })
    )?
    .label("Target")
    .legend(|(x, y)| Circle::new((x, y), 5, BLUE.filled()));

    // Original source points (赤)
    chart.draw_series(
        original_source.rows().into_iter().map(|row| {
            Circle::new((row[0], row[1]), 5, RED.filled())
        })
    )?
    .label("Original Source")
    .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    // Current source points (緑)
    chart.draw_series(
        current_source.rows().into_iter().map(|row| {
            Circle::new((row[0], row[1]), 5, GREEN.filled())
        })
    )?
    .label("Current Source")
    .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Plot saved to {}", filename);
    Ok(())
}