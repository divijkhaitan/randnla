use std::error::Error;
#[derive(Debug)]
pub enum RandNLAError {
    InvalidParameters(String),
    InvalidDimensions(String),
    NegativeDimensions(String),
    NotOverdetermined(String),
    NotSquare(String),
    SingularMatrix(String),
}

impl std::fmt::Display for RandNLAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RandNLAError::InvalidParameters(msg) | 
            RandNLAError::InvalidDimensions(msg) | 
            RandNLAError::NegativeDimensions(msg) | 
            RandNLAError::NotOverdetermined(msg) |
            RandNLAError::SingularMatrix(msg) |
            RandNLAError::NotSquare(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for RandNLAError {}
