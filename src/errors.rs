use std::error::Error;
#[derive(Debug)]
pub enum DimensionError {
    InvalidDimensions(String),
    NegativeDimensions(String),
    NotOverdetermined(String),
}

impl std::fmt::Display for DimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimensionError::InvalidDimensions(msg) | 
            DimensionError::NegativeDimensions(msg) | 
            DimensionError::NotOverdetermined(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for DimensionError {}
