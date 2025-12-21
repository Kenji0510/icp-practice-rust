use ndarray::prelude::*;
use anyhow::{Result};
use pcd_rs::{PcdDeserialize, PcdSerialize, Reader, WriterInit};


#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZT {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone)]
pub struct Points {
    pub points: Vec<PointXYZ>,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZRGB {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rgb: f32,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
}

pub fn rgb_to_float(r: u8, g: u8, b: u8) -> f32 {
    let rgb_int: u32 = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
    f32::from_bits(rgb_int)
}

pub fn save_pcd_with_normals(
    points: &[PointXYZNormal],
    file_path: &str,
) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;
    
    for point in points {
        writer.push(point)?;
    }
    
    writer.finish()?;
    Ok(())
}

pub fn save_pcd(
    points: &[PointXYZRGB],
    file_path: &str,
) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;
    
    for point in points {
        writer.push(point)?;
    }
    writer.finish()?;

    Ok(())
}


impl Points {
    pub fn new(p: Vec<PointXYZ>) -> Self {
        Points {
            points: p,
        }
    }

    pub fn apply_transform(&mut self, transform: &Array2<f64>) {
        assert_eq!(transform.shape(), &[4, 4], "Transform must be 4x4 matrix");

        let transformed_points: Vec<PointXYZ> = self.points.iter()
            .map(|p| {
                // 同次座標に変換 [x, y, z, 1]
                let point_homogeneous = array![p.x as f64, p.y as f64, p.z as f64, 1.0];
                
                // 変換適用: T * p
                let transformed = transform.dot(&point_homogeneous);
                
                // 同次座標から3D座標に戻す
                PointXYZ {
                    x: transformed[0] as f32,
                    y: transformed[1] as f32,
                    z: transformed[2] as f32,
                }
            })
            .collect();

        self.points = transformed_points;
    }

    pub fn transform_colored_points(
        &self,
        color: (u8, u8, u8),
    ) -> Vec<PointXYZRGB> {
        let pts = self.points.iter()
            .map(|p| PointXYZRGB {
                x: p.x,
                y: p.y,
                z: p.z,
                rgb: rgb_to_float(color.0, color.1, color.2)
            })
            .collect();

        pts
    }

    pub fn save_pcd(
        &self,
        file_path: &str,
        color: (u8, u8, u8),
    ) -> Result<()> {
        let colored_points: Vec<PointXYZRGB> = self.points.iter()
            .map(|p| PointXYZRGB {
                x: p.x,
                y: p.y,
                z: p.z,
                rgb: rgb_to_float(color.0, color.1, color.2)
            })
            .collect();

        let mut writer = pcd_rs::WriterInit {
            width: 1,
            height: colored_points.len() as u64,
            viewpoint: Default::default(),
            data_kind: pcd_rs::DataKind::Ascii,
            schema: None,
        }
        .create(file_path)?;
        
        for point in &colored_points {
            writer.push(point)?;
        }
        writer.finish()?;

        Ok(())
    }

    pub fn save_pcd_xyz(
        &self,
        file_path: &str,
    ) -> Result<()> {
        let colored_points: Vec<PointXYZ> = self.points.iter()
            .map(|p| PointXYZ {
                x: p.x,
                y: p.y,
                z: p.z,
            })
            .collect();

        let mut writer = pcd_rs::WriterInit {
            width: 1,
            height: colored_points.len() as u64,
            viewpoint: Default::default(),
            data_kind: pcd_rs::DataKind::Ascii,
            schema: None,
        }
        .create(file_path)?;
        
        for point in &colored_points {
            writer.push(point)?;
        }
        writer.finish()?;

        Ok(())
    }
}


pub fn load_pcd_xyz(
    file_path: &str,
) -> Result<Vec<PointXYZ>> {
    let reader = match Reader::open(file_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open PCD file: {}", e);
            return Err(anyhow::anyhow!("Failed to open PCD file: {}", e));
        }
    };

    let points: Vec<PointXYZ> = match reader.collect() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to read PCD data: {}", e);
            return Err(anyhow::anyhow!("Failed to read PCD data: {}", e));
        }
    };

    Ok(points)
}

pub fn load_pcd_xyzt(
    file_path: &str,
) -> Result<Vec<PointXYZT>> {
    let reader = match Reader::open(file_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open PCD file: {}", e);
            return Err(anyhow::anyhow!("Failed to open PCD file: {}", e));
        }
    };

    let points: Vec<PointXYZT> = match reader.collect() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to read PCD data: {}", e);
            return Err(anyhow::anyhow!("Failed to read PCD data: {}", e));
        }
    };

    Ok(points)
}