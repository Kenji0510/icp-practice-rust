use ndarray::{Array2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use rustc_hash::FxHasher; // 高速なハッシャー推奨

// FastMapの型定義（FxHashMap推奨）
type FastMap<V> = HashMap<u64, V, BuildHasherDefault<FxHasher>>;

#[derive(Default)]
struct VoxelStat {
    sum: [f64; 3],
    count: usize,
}

impl VoxelStat {
    // マージ用関数
    fn merge(&mut self, other: &VoxelStat) {
        self.sum[0] += other.sum[0];
        self.sum[1] += other.sum[1];
        self.sum[2] += other.sum[2];
        self.count += other.count;
    }
}

#[inline]
fn morton3d(ix: u32, iy: u32, iz: u32) -> u64 {
    fn part1by2(n: u32) -> u64 {
        // 21bit制限を超えないように念のためマスク（元のコード通り）
        let mut x = n as u64 & 0x1fffff;
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
    if n_points == 0 {
        return Array2::zeros((0, 3));
    }

    // 1. 最小値・最大値の計算 (並列化で高速化)
    // ndarrayのaxis_iterを使うと安全
    let (min_corner, max_corner) = points.axis_iter(Axis(0))
        .into_par_iter()
        .fold(
            || (vec![f64::INFINITY; 3], vec![f64::NEG_INFINITY; 3]),
            |(mut min_acc, mut max_acc), row| {
                for i in 0..3 {
                    min_acc[i] = min_acc[i].min(row[i]);
                    max_acc[i] = max_acc[i].max(row[i]);
                }
                (min_acc, max_acc)
            }
        )
        .reduce(
            || (vec![f64::INFINITY; 3], vec![f64::NEG_INFINITY; 3]),
            |(min_a, max_a), (min_b, max_b)| {
                (
                    vec![min_a[0].min(min_b[0]), min_a[1].min(min_b[1]), min_a[2].min(min_b[2])],
                    vec![max_a[0].max(max_b[0]), max_a[1].max(max_b[1]), max_a[2].max(max_b[2])]
                )
            }
        );

    // 安全対策: グリッド数が21bit(約200万)を超えないかチェック
    // 超える場合はボクセルサイズを強制調整するか、エラーにするか、あきらめる
    let inv = 1.0 / voxel_size;
    let max_idx_x = ((max_corner[0] - min_corner[0]) * inv) as u64;
    let max_idx_y = ((max_corner[1] - min_corner[1]) * inv) as u64;
    let max_idx_z = ((max_corner[2] - min_corner[2]) * inv) as u64;
    
    if max_idx_x > 0x1fffff || max_idx_y > 0x1fffff || max_idx_z > 0x1fffff {
        eprintln!("Warning: Point cloud extent exceeds Morton code limit (21-bit). Results may collide.");
        // 実戦ではここで return Err したり、voxel_size を大きくしたりする
    }

    // 2. Map-Reduce パターンによるボクセル集計
    let global_map = points.axis_iter(Axis(0))
        .into_par_iter()
        .fold(
            FastMap::default, // 各スレッドの初期値
            |mut map: FastMap<VoxelStat>, row| {
                let x = row[0];
                let y = row[1];
                let z = row[2];

                // 安全なインデックス計算: max(0.0) でマイナス排除
                let ix = ((x - min_corner[0]) * inv).max(0.0) as u32;
                let iy = ((y - min_corner[1]) * inv).max(0.0) as u32;
                let iz = ((z - min_corner[2]) * inv).max(0.0) as u32;
                
                let key = morton3d(ix, iy, iz);
                let stat = map.entry(key).or_default();
                stat.sum[0] += x;
                stat.sum[1] += y;
                stat.sum[2] += z;
                stat.count += 1;
                map
            }
        )
        .reduce(
            FastMap::default, // マージ処理
            |mut map_a, map_b| {
                for (k, v) in map_b {
                    map_a.entry(k).or_default().merge(&v);
                }
                map_a
            }
        );

    // 3. 結果の生成
    let downsampled_count = global_map.len();
    let mut result = Array2::<f64>::zeros((downsampled_count, 3));
    
    for (i, stat) in global_map.values().enumerate() {
        let count_f = stat.count as f64;
        result[[i, 0]] = stat.sum[0] / count_f;
        result[[i, 1]] = stat.sum[1] / count_f;
        result[[i, 2]] = stat.sum[2] / count_f;
    }

    result
}