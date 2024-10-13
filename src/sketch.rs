use nalgebra::DMatrix;
use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};
use rand::thread_rng;
use crate::errors::DimensionError;
use std::error::Error;
pub enum DistributionType {
    Gaussian,
    Uniform,
    Rademacher,
}

pub enum MatrixAttribute {
    Row,
    Column,
}

// Haar Sample with error handling
pub fn haar_sample(rows: usize, columns: usize, attr: MatrixAttribute) -> Result<DMatrix<f64>, Box<dyn Error>> {
    if rows == 0 || columns == 0 {
        return Err(Box::new(DimensionError::InvalidDimensions(
            "Dimensions cannot be zero".to_string(),
        )));
    }

    let (m, n) = match attr {
        MatrixAttribute::Row => {
            if rows > columns {
                return Err(Box::new(DimensionError::InvalidDimensions(format!(
                    "Cannot have more rows ({}) than columns ({}) for row-orthonormal matrix",
                    rows, columns
                ))));
            }
            (columns, rows)
        }
        MatrixAttribute::Column => {
            if columns > rows {
                return Err(Box::new(DimensionError::InvalidDimensions(format!(
                    "Cannot have more columns ({}) than rows ({}) for column-orthonormal matrix",
                    columns, rows
                ))));
            }
            (rows, columns)
        }
    };

    let mut rng = thread_rng();
    let normal = StandardNormal.sample_iter(&mut rng);
    let data: Vec<f64> = normal.take(m * n).collect();

    let matrix = DMatrix::from_vec(m, n, data);
    let (mut q, r) = matrix.qr().unpack();
    for i in 0..q.ncols() {
        // Get the sign of the diagonal element from r
        let sign = r[(i, i)].signum();

        // Scale the i-th column of q by this sign
        q.set_column(i, &(&q.column(i) * sign));
    }
    match attr {
        MatrixAttribute::Row => Ok(q.transpose()),
        MatrixAttribute::Column => Ok(q),
    }
}

// Sketching Operator with error handling
pub fn sketching_operator(
    dist_type: DistributionType, 
    rows: usize, 
    cols: usize
) -> Result<DMatrix<f64>, Box<dyn Error>> {
    if rows == 0 || cols == 0 {
        return Err(Box::new(DimensionError::InvalidDimensions(
            "Rows and columns must be greater than 0".to_string(),
        )));
    }

    let mut rng = thread_rng();
    let matrix = match dist_type {
        DistributionType::Gaussian => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            DMatrix::from_fn(rows, cols, |_i, _j| normal.sample(&mut rng))
        },
        DistributionType::Uniform => {
            let uniform = Uniform::new(-1.0, 1.0);
            DMatrix::from_fn(rows, cols, |_i, _j| uniform.sample(&mut rng))
        },
        DistributionType::Rademacher => {
            let bernoulli = Bernoulli::new(0.5).unwrap();
            DMatrix::from_fn(rows, cols, |_i, _j| if bernoulli.sample(&mut rng) { 1.0 } else { -1.0 })
        }
    };

    Ok(matrix)
}


#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use super::{MatrixAttribute, haar_sample, sketching_operator, DistributionType};
    use rand::Rng;
    
    #[test]
    fn test_haar_sample_row_attribute() {
        let m = rand::thread_rng().gen_range(50..100);
        let n = rand::thread_rng().gen_range(m..500);
        
        // Case 1: m < n (should not return an error)
        let result = haar_sample(m, n, MatrixAttribute::Row);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.nrows(), m);
        assert_eq!(result.ncols(), n);
        
        // Normal rows
        for i in 0..m {
            assert_relative_eq!(result.row(i).norm(), 1.0, epsilon = 1e-6);
        }
        
        // Orthogonal rows
        for i in 0..m {
            for j in (i+1)..m {
                assert_relative_eq!(result.row(i).dot(&result.row(j)), 0.0, epsilon = 1e-6);
            }
        }
    
        // Case 2: m > n (should return an error)
        let n = rand::thread_rng().gen_range(50..100);
        let m = rand::thread_rng().gen_range(n..500);
        let result = haar_sample(m, n, MatrixAttribute::Row);
        assert!(result.is_err());
    }

    #[test]
    fn test_haar_sample_column_attribute() {
        let n = rand::thread_rng().gen_range(100..150);
        let m = rand::thread_rng().gen_range(500..5000);
        
        // Case 3: m > n (should not return an error)
        let result = haar_sample(m, n, MatrixAttribute::Column);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.nrows(), m);
        assert_eq!(result.ncols(), n);
        
        // Normal columns
        for j in 0..n {
            assert_relative_eq!(result.column(j).norm(), 1.0, epsilon = 1e-6);
        }

        // Orthogonal columns
        for i in 0..n {
            for j in (i+1)..n {
                assert_relative_eq!(result.column(i).dot(&result.column(j)), 0.0, epsilon = 1e-6);
            }
        }

        // Case 4: m < n (should return an error)
        let m = rand::thread_rng().gen_range(50..100);
        let n = rand::thread_rng().gen_range(m..500);
        let result = haar_sample(m, n, MatrixAttribute::Column);
        assert!(result.is_err());
    }

    #[test]
    fn test_sketching_operator() {
        let rows = 100;
        let cols = 50;

        // Test Gaussian distribution
        let result = sketching_operator(DistributionType::Gaussian, rows, cols);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.nrows(), rows);
        assert_eq!(result.ncols(), cols);

        // Test Uniform distribution
        let result = sketching_operator(DistributionType::Uniform, rows, cols);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.nrows(), rows);
        assert_eq!(result.ncols(), cols);

        // Test Rademacher distribution
        let result = sketching_operator(DistributionType::Rademacher, rows, cols);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.nrows(), rows);
        assert_eq!(result.ncols(), cols);

        // Edge case: Zero dimensions (should return an error)
        let result = sketching_operator(DistributionType::Gaussian, 0, cols);
        assert!(result.is_err());

        let result = sketching_operator(DistributionType::Gaussian, rows, 0);
        assert!(result.is_err());
    }
}
