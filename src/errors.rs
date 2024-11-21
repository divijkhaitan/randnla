use std::error::Error;
#[derive(Debug)]
pub enum RandNLAError {
    InvalidParameters(String),
    InvalidDimensions(String),
    NegativeDimensions(String),
    NotOverdetermined(String),
    NotSquare(String),
    SingularMatrix(String),
    MatrixDecompositionError(String),
    NotHermitian(String),
    NotPositiveSemiDefinite(String),
    ComputationError(String),
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
            RandNLAError::MatrixDecompositionError(msg) => write!(f, "Matrix decomposition error: {}", msg),
            RandNLAError::NotHermitian(msg) => write!(f, "Not a Hermitian matrix: {}", msg),
            RandNLAError::NotPositiveSemiDefinite(msg) => write!(f, "Not a positive semi-definite matrix: {}", msg),
            RandNLAError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl Error for RandNLAError {}
