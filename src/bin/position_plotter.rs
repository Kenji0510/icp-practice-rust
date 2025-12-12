use anyhow::{Context, Result};
use plotters::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

// JSONの構造に合わせたデータ定義 (読み込み用)
#[derive(Debug, Deserialize)]
struct PoseData {
    timestamp: f64,
    position: [f32; 3],      // x, y, z
    rotation_quat: [f32; 4], // w, x, y, z
}

#[derive(Debug, Deserialize)]
struct TrajectoryOutput {
    icp_trajectory: Vec<PoseData>,
    imu_raw_trajectory: Vec<PoseData>,
}

fn main() -> Result<()> {
    // 1. JSONデータの読み込み
    let input_path = "data/output/icp_map/myself-position/trajectory_comparison-20251125-06.json";
    let output_image_path = "data/output/icp_map/trajectory_plot-20251125-06.png";

    println!("Loading trajectory data from {}...", input_path);
    let file = File::open(input_path).context("Failed to open JSON file")?;
    let reader = BufReader::new(file);
    let data: TrajectoryOutput = serde_json::from_reader(reader)?;

    println!("ICP points: {}", data.icp_trajectory.len());
    println!("IMU points: {}", data.imu_raw_trajectory.len());

    // 2. 描画データの準備 ( (x, y) のタプルに変換 )
    let icp_series: Vec<(f32, f32)> = data.icp_trajectory.iter()
        .map(|p| (p.position[0], p.position[1]))
        .collect();

    let imu_series: Vec<(f32, f32)> = data.imu_raw_trajectory.iter()
        .map(|p| (p.position[0], p.position[1]))
        .collect();

    if icp_series.is_empty() {
        return Err(anyhow::anyhow!("Trajectory data is empty"));
    }

    // 3. グラフの範囲（Bounding Box）を決定
    // ICPとIMUの両方が収まるように最大最小を探す
    // let all_points: Vec<&(f32, f32)> = icp_series.iter().chain(imu_series.iter()).collect();
    let all_points: Vec<&(f32, f32)> = icp_series.iter().collect();
    
    let min_x = all_points.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let max_x = all_points.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
    let min_y = all_points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_y = all_points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);

    let data_w = max_x - min_x;
    let data_h = max_y - min_y;
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;

    // --- 2. 画像のアスペクト比に合わせて表示範囲を補正 (ここが修正の肝) ---
    // 画像サイズ
    let img_w = 1024.0;
    let img_h = 768.0;
    let img_aspect = img_w / img_h; // 約 1.33

    // 現在のデータの縦横比
    // 0除算回避のため max(..., 1e-6)
    let data_aspect = data_w / data_h.max(1e-6);

    let (view_w, view_h) = if data_aspect > img_aspect {
        // データの方が「横長」の場合 (今回のケースはこれに該当するはず)
        // 横幅を基準にして、高さを画像のアスペクト比に合わせて広げる
        let w = data_w;
        let h = w / img_aspect;
        (w, h)
    } else {
        // データの方が「縦長」または正方形に近い場合
        // 高さを基準にして、横幅を画像のアスペクト比に合わせて広げる
        let h = data_h;
        let w = h * img_aspect;
        (w, h)
    };

    // 余白を少し追加 (10%)
    let margin_ratio = 1.1;
    let final_view_w = view_w * margin_ratio;
    let final_view_h = view_h * margin_ratio;

    let x_range = (center_x - final_view_w / 2.0)..(center_x + final_view_w / 2.0);
    let y_range = (center_y - final_view_h / 2.0)..(center_y + final_view_h / 2.0);

    println!("Data Range: X={:.2}..{:.2} (W={:.2}), Y={:.2}..{:.2} (H={:.2})", min_x, max_x, data_w, min_y, max_y, data_h);
    println!("Plot Range: X={:?}, Y={:?}", x_range, y_range);

    // 4. 描画バックエンドのセットアップ
    let root = BitMapBackend::new(output_image_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Trajectory Comparison: ICP vs IMU Raw", ("sans-serif", 40).into_font())
        // .caption("Trajectory Comparison: ICP Raw", ("sans-serif", 40).into_font())
        // .caption("Trajectory Comparison: IMU Raw", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh()
        .x_desc("X Position (m)")
        .y_desc("Y Position (m)")
        .light_line_style(&WHITE) // グリッドを見やすく
        .draw()?;

    // 5. データの描画
    // IMU Raw Trajectory (青色)
    chart
        .draw_series(LineSeries::new(
            imu_series,
            &BLUE,
        ))?
        .label("IMU Raw Integration")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // ICP Trajectory (赤色)
    chart
        .draw_series(LineSeries::new(
            icp_series,
            &RED,
        ))?
        .label("ICP SLAM")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        
    // 開始点にマーカーを打つ
    if let Some(start) = data.icp_trajectory.first() {
        chart.draw_series(PointSeries::of_element(
            vec![(start.position[0], start.position[1])],
            5,
            &BLACK,
            &|c, s, st| {
                return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
                + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
                + Text::new("Start", (10, 0), ("sans-serif", 15).into_font());
            },
        ))?;
    }

    // 凡例の描画
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // 6. 画像として保存
    root.present()?;
    println!("Plot saved to {}", output_image_path);

    Ok(())
}