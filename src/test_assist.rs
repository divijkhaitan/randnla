use nalgebra::{DMatrix, DVector, dmatrix, dvector};
use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};use rand::thread_rng;



/// Generates a random matrix of size (rows, cols) with normally distributed elems
pub fn generate_random_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| normal.sample(&mut rng)).collect();
    DMatrix::from_vec(rows, cols, data)
}

/// Generates a random Hermitian matrix of size (n, n) with normally distributed elems
pub fn generate_random_hermitian_matrix(n: usize) -> DMatrix<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..n * n).map(|_| normal.sample(&mut rng)).collect();
    let mut matrix = DMatrix::from_vec(n, n, data);
    let matrix_t = matrix.transpose();
    matrix = 0.5 * (matrix + matrix_t); // symmetric
    matrix
}

/// Generates a random psd matrix of size (n, n) with normally distributed elems
pub fn generate_random_psd_matrix(n: usize) -> DMatrix<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..n * n).map(|_| normal.sample(&mut rng)).collect();
    let mut matrix = DMatrix::from_vec(n, n, data);
    let matrix_t = matrix.transpose();
    matrix = matrix * matrix_t; 
    matrix
}

pub fn check_approx_equal(a: &DMatrix<f64>, b: &DMatrix<f64>, tolerance: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if (a[(i, j)] - b[(i, j)]).abs() > tolerance {
                // println!("{}, {}, {}, {}", i, j, a[(i, j)], b[(i, j)]);
                return false;
            }
        }
    }
    
    true
}