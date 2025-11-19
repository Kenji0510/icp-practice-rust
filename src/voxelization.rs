use ndarray::Array2;
use rayon::prelude::*;
use std::{collections::HashMap, time::Instant};

use nohash_hasher::NoHashHasher;

#[derive(Default, Debug)]
struct VoxelStat {
    sum: [f64; 3],
    count: u32,
}

type FastMap<V> = HashMap<u64, V, std::hash::BuildHasherDefault<NoHashHasher<u64>>>;

#[inline]
fn morton3d(ix: u32, iy: u32, iz: u32) -> u64 {
    fn part1by2(n: u32) -> u64 {
        // 21bit → 63bit へ 0bit 埋め込み（マジックビット法）
        let mut x = n as u64 & 0x1fffff; // 21 bit
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        x
    }
    part1by2(ix) | (part1by2(iy) << 1) | (part1by2(iz) << 2)
}

pub fn voxel_downsample_array2(points: &Array2<f64>, voxel_size: f64) -> Array2<f64> {
    let n_points = points.nrows();
    
    // 最小座標を計算
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    
    for i in 0..n_points {
        min_x = min_x.min(points[[i, 0]]);
        min_y = min_y.min(points[[i, 1]]);
        min_z = min_z.min(points[[i, 2]]);
    }

    let inv = 1.0 / voxel_size;

    // 並列処理: チャンクごとに局所的なハッシュマップを作成
    let chunk_size = 10_000;
    let thread_maps: Vec<FastMap<VoxelStat>> = (0..n_points)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            let mut map: FastMap<VoxelStat> = FastMap::default();
            for i in chunk {
                let x = points[[i, 0]];
                let y = points[[i, 1]];
                let z = points[[i, 2]];
                
                let ix = ((x - min_x) * inv).floor() as u32;
                let iy = ((y - min_y) * inv).floor() as u32;
                let iz = ((z - min_z) * inv).floor() as u32;
                let key = morton3d(ix, iy, iz);
                
                let stat = map.entry(key).or_default();
                stat.sum[0] += x;
                stat.sum[1] += y;
                stat.sum[2] += z;
                stat.count += 1;
            }
            map
        })
        .collect();

    // グローバルマップにマージ
    let mut global: FastMap<VoxelStat> = FastMap::default();
    for local in thread_maps {
        for (k, v) in local {
            let g = global.entry(k).or_default();
            g.sum[0] += v.sum[0];
            g.sum[1] += v.sum[1];
            g.sum[2] += v.sum[2];
            g.count += v.count;
        }
    }

    // 結果を Array2 に変換
    let downsampled_count = global.len();
    let mut result = Array2::<f64>::zeros((downsampled_count, 3));
    
    for (i, stat) in global.values().enumerate() {
        let count_f = stat.count as f64;
        result[[i, 0]] = stat.sum[0] / count_f;
        result[[i, 1]] = stat.sum[1] / count_f;
        result[[i, 2]] = stat.sum[2] / count_f;
    }

    result
}