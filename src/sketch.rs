use nalgebra::DMatrix;
use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};
use rand::thread_rng;

pub enum DistributionType {
    Gaussian,
    Uniform,
    Rademacher,
}

pub enum MatrixAttribute {
    Row,
    Column,
}

pub fn haar_sample(rows: usize, columns: usize, attr: MatrixAttribute) -> DMatrix<f64> {
    // Ensuring valid matrix dimensions, an orthonormal matrix cannot have 
    let (m, n) = match attr {
        MatrixAttribute::Row => {
            if rows > columns
            {
                panic!("Cannot have more rows than columns for row-orthonormal matrix, given {} rows and {} columns", rows, columns);
            }
                (columns, rows)
        }
        MatrixAttribute::Column => {
            if columns > rows
            {
                panic!("Cannot have more columns than rows for column-orthonormal matrix, given {} columns and {} rows", columns, rows);
            }
        (rows, columns)

        }
    };

    let mut rng = thread_rng();
    let normal = StandardNormal.sample_iter(&mut rng);
    let data: Vec<f64> = normal.take(m * n).collect();

    let matrix = DMatrix::from_vec(m, n, data);
    let (q, _) = matrix.qr().unpack();

    match attr {
        MatrixAttribute::Row => q.transpose(),
        MatrixAttribute::Column => q,
    }
}

pub fn sketching_operator(
    dist_type: DistributionType, 
    rows: usize, 
    cols: usize
) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
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
    matrix
}


#[cfg(test)]
mod tests
{
    use approx::assert_relative_eq;
    use super::{MatrixAttribute, haar_sample};
    use rand::Rng;
    use super::DMatrix;
    fn sum_of_squares_row(matrix: &DMatrix<f64>, row: usize) -> f64 {
        matrix.row(row).dot(&matrix.row(row))
    }
    
    fn sum_of_squares_column(matrix: &DMatrix<f64>, col: usize) -> f64 {
        matrix.column(col).dot(&matrix.column(col))
    }
    
    #[test]
    fn test_row_attribute() {
        let m = rand::thread_rng().gen_range(50..100);
        let n = rand::thread_rng().gen_range(m..500);
        
        // Case 1: m < n (should not panic)
        let result = haar_sample(m, n, MatrixAttribute::Row);
        assert_eq!(result.nrows(), m);
        assert_eq!(result.ncols(), n);
        
        // Normal rows
        for i in 0..m {
            assert_relative_eq!(sum_of_squares_row(&result, i), 1.0, epsilon = 1e-6);
        }
        
        // Orthogonal rows
        for i in 0..m {
            for j in (i+1)..m{
            assert_relative_eq!(result.row(i).dot(&result.row(j)), 0.0, epsilon = 1e-6);
            }
        }
    
        // Columns have magnitude at most 1
        for j in 0..n {
            let sum = sum_of_squares_column(&result, j);
            assert!(sum >= 0.0 && sum <= 1.0);
        }
        
        // Case 2: m > n (should panic)
        let n = rand::thread_rng().gen_range(50..100);
        let m = rand::thread_rng().gen_range(n..500);
        let result = std::panic::catch_unwind(|| haar_sample(m, n, MatrixAttribute::Row));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_column_attribute() {
        let n = rand::thread_rng().gen_range(50..100);
        let m = rand::thread_rng().gen_range(n..500);
        
        // Case 3: m > n (should not panic)
        let result = haar_sample(m, n, MatrixAttribute::Column);
        assert_eq!(result.nrows(), m);
        assert_eq!(result.ncols(), n);
        
        // Normal Columns
        for j in 0..n {
            assert_relative_eq!(sum_of_squares_column(&result, j), 1.0, epsilon = 1e-6);
        }
        // Orthogonal Columns
        for i in 0..n {
            for j in (i+1)..n{
                assert_relative_eq!(result.column(i).dot(&result.column(j)), 0.0, epsilon = 1e-6);
            }
        }
        
        // Rows have magnitude at most 1
        for i in 0..m {
            let sum = sum_of_squares_row(&result, i);
            assert!(sum >= 0.0 && sum <= 1.0);
        }
        
        // Case 4: m < n (should panic)
        let m = rand::thread_rng().gen_range(50..100);
        let n = rand::thread_rng().gen_range(m..500);
        let result = std::panic::catch_unwind(|| haar_sample(m, n, MatrixAttribute::Column));
        assert!(result.is_err());
    }
}