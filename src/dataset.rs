use std::{error::Error, path::Path, fs::File};


pub trait DatasetInfo {
    fn name(&self) -> Option<String> {
        None
    }
    fn authors(&self) -> Option<String> {
        None
    }
    fn instructions(&self) -> Option<String> {
        None
    }
    fn published(&self) -> Option<String> {
        None
    }
    fn last_updated(&self) -> Option<String> {
        None
    }
}

pub trait LoadDataset<LoadOptions>: Sized {
    fn load(options: LoadOptions) -> Result<Self, Box<dyn Error>>;
}

pub  trait IntoTensors {

}

pub trait Dataset<LoadOptions>: LoadDataset<LoadOptions> + DatasetInfo {
}

pub trait LoadDatasetFromFile: Sized
where
    for<'a> Self: serde::Deserialize<'a>,
{
    
}

impl<TOpt: AsRef<Path>, T: LoadDatasetFromFile> LoadDataset<TOpt> for T {
    fn load(options: TOpt) -> Result<Self, Box<dyn Error>> {
        Ok(serde_json::from_reader(std::io::BufReader::new(
            File::open(options)?,
        ))?)
    }
}
