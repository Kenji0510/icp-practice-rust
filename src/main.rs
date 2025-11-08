use anyhow::{Result, Context};
use icp_practice::operate_pcd::{PointXYZ, Points, load_pcd_xyz, save_pcd};
use kdtree::{KdTree, distance::squared_euclidean};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{Determinant, SVD};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// const WIDTH: f64 = 10.0;
// const HEIGHT: f64 = 5.0;
// const ROTATION_ANGLE_DEG: f64 = 25.0;
// const TRANSLATION_X: f64 = 5.0;
// const TRANSLATION_Y: f64 = 3.0;
// const NOISE_LEVEL: f64 = 0.1; // ノイズを少し強めに
const SAMPLE_SIZE: usize = 300;
const TRIM_PERCENTAGE: f64 = 0.9;

fn main() -> Result<()> {
    let target_pcd_file_path = "data/input/removed-ceiling-output-025.pcd";
    let source_pcd_file_path = "data/input/removed-ceiling-cloud_registered_body_0_025.pcd";
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
    let transform_matrix = array![
        [0.962047, 0.259369, -0.084812, 2.365178],
        [-0.272116, 0.888537, -0.369399, 0.685336],
        [-0.020452, 0.378457, 0.925393, 0.015746],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ];
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
    source_pts.apply_transform(&transform_matrix);

    let target_pts_arr = points_to_array2(&target_pts);
    let source_pts_arr = points_to_array2(&source_pts);

    let max_iterations = 20;
    let tolerance = 1e-1;

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

    let start = std::time::Instant::now();
    for i in 0..max_iterations {
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

        // Find closest points
        let (matched_target_pts, distance_sq) = find_closest_pairs_kdtree(&sampled_source_pts, &target_pts_arr, &kdtree);

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

        let inlier_source_pts = sampled_source_pts.select(Axis(0), &inlier_indices);
        let inlier_target_pts = matched_target_pts.select(Axis(0), &inlier_indices);

        let (R, t) = calculate_transformation(&inlier_source_pts, &inlier_target_pts);

        current_source_pts_arr = current_source_pts_arr.dot(&R.t()) + &t;

        let transformed_inlier_pts = current_source_pts_arr.select(Axis(0), &sampled_indices)
                                                            .select(Axis(0), &inlier_indices);
        let current_error = calculate_mean_error(
            &transformed_inlier_pts, 
            &inlier_target_pts
        );
        
        println!("Iteration {}: mean error (from {} inliers, {:.0}% kept) = {}", 
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

    println!("Final aligned source points:\n{:?}", current_source_pts_arr);

    // plot_points(&source_pts, &target_pts, &current_source_pts, "icp_final.png", "Final State").unwrap();

    let aligned_source_pts = array2_to_points(&current_source_pts_arr);

    let colored_target_pts = target_pts.transform_colored_points((0, 0, 255)); // 青
    let colored_source_pts = source_pts.transform_colored_points((255, 0, 0)); // 赤
    let colored_aligned_source_pts = aligned_source_pts.transform_colored_points((0, 255, 0)); // 緑

    let mut all_points = colored_target_pts.clone();
    all_points.extend(colored_aligned_source_pts.clone());
    all_points.extend(colored_source_pts.clone());

    // Save each point clouds
    let save_path = "data/output/icp_aligned_result_v-025.pcd";
    match save_pcd(&all_points, save_path) {
        Ok(_) => println!("Saved aligned points to {}", save_path),
        Err(e) => eprintln!("Failed to save PCD file: {}", e),
    }

    Ok(())
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
) -> (Array2<f64>, Vec<f64>) {
    
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
    (matched_target_pts, distance_sq)
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