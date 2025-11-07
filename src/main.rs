use anyhow::{Result, Context};
use icp_practice::operate_pcd::{PointXYZ, Points, load_pcd_xyz, save_pcd};
use plotters::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::{Determinant, SVD};

const WIDTH: f64 = 10.0;
const HEIGHT: f64 = 5.0;
const ROTATION_ANGLE_DEG: f64 = 25.0;
const TRANSLATION_X: f64 = 5.0;
const TRANSLATION_Y: f64 = 3.0;
// const NOISE_LEVEL: f64 = 0.1; // ノイズを少し強めに

fn main() -> Result<()> {
    let target_pcd_file_path = "data/input/clipped_Laser_map_5_voxel-01.pcd";
    let source_pcd_file_path = "data/input/clipped_rotated_Laser_map_5_voxel-01.pcd";
    let target_d = load_pcd_xyz(target_pcd_file_path)
        .context("Failed to load PCD file")?;
    let source_d = load_pcd_xyz(source_pcd_file_path)
        .context("Failed to load PCD file")?;

    let target_pts = Points::new(target_d);
    println!("Loaded {} points from {}", target_pts.points.len(), target_pcd_file_path);

    let mut source_pts = Points::new(source_d);
    println!("Loaded {} points from {}", source_pts.points.len(), source_pcd_file_path);

    let transform_matrix = array![
        [0.959326, 0.282294, -0.002065, 2.249126],
        [-0.282291, 0.959327, 0.001695, 0.171887],
        [0.002459, -0.001043, 0.999996, -0.001295],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ];
    source_pts.apply_transform(&transform_matrix);

    let target_pts_arr = points_to_array2(&target_pts);
    let source_pts_arr = points_to_array2(&source_pts);

    // let target_pts = array![
    //     [0., 0.],
    //     [10., 0.],
    //     [10., 5.],
    //     [0., 5.]
    // ];
    // println!("Target points: {:?}", target_pts);
    
    // let angle_rad = ROTATION_ANGLE_DEG.to_radians();
    // let R = array![
    //     [angle_rad.cos(), -angle_rad.sin()],
    //     [angle_rad.sin(), angle_rad.cos()]
    // ];
    // let t = array![TRANSLATION_X, TRANSLATION_Y];
    // let source_pts = target_pts.dot(&R.t()) + &t;
    // println!("Source points: {:?}", source_pts);

    // plot_points(&source_pts, &target_pts, &source_pts, "icp_initial.png", "Initial State").unwrap();

    let max_iterations = 20;
    let tolerance = 1e-5;

    let mut current_source_pts_arr = source_pts_arr.clone();

    let start = std::time::Instant::now();
    for i in 0..max_iterations {
        // Find closest points
        let (matched_target_pts, _) = find_closest_pairs(&current_source_pts_arr, &target_pts_arr);

        let (R, t) = calculate_transformation(&current_source_pts_arr, &matched_target_pts);

        current_source_pts_arr = current_source_pts_arr.dot(&R.t()) + &t;

        let current_error = calculate_mean_error(&current_source_pts_arr, &matched_target_pts);
        println!("Iteration {}: mean error = {}", i + 1, current_error);

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
    let save_path = "data/output/icp_aligned_result.pcd";
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