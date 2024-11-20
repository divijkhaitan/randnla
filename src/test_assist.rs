use nalgebra::{DMatrix, DVector, dmatrix, dvector};
use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};use rand::thread_rng;



/// Generates a random matrix of size (rows, cols) with normally distributed entries using thread_rng

pub fn generate_random_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| normal.sample(&mut rng)).collect();
    DMatrix::from_vec(rows, cols, data)
}