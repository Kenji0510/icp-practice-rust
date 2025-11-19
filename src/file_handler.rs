use std::{fs, path::PathBuf};

use anyhow::{Context, Result};


pub fn load_pcd_files(
    dir_path: &str,
) -> Result<Vec<PathBuf>> {
    let re = regex::Regex::new(r"voxelized-025_frame_(\d+)\.pcd$")
        .context("Invalid regex pattern")?;

    let entries = fs::read_dir(dir_path)
        .context(format!("Failed to read directory: {}", dir_path))?;

    let mut files_with_numbers: Vec<(PathBuf, u32)> = Vec::new();

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => continue,
        };

        if let Some(captures) = re.captures(filename) {
            if let Some(num_str) = captures.get(1) {
                if let Ok(num) = num_str.as_str().parse::<u32>() {
                    files_with_numbers.push((path.clone(), num));
                }
            }
        }
    }

    files_with_numbers.sort_by_key(|(_path, num)| *num);

    let sorted_paths: Vec<PathBuf> = files_with_numbers
        .into_iter()
        .map(|(path, _num)| path)
        .collect();

    Ok(sorted_paths)
}