// mod sample;
mod cg;

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::time::Instant;
use sketch::haar_sample;

mod sketch;
mod sketch_and_solve;
mod solvers;
mod sketch_and_precondition;
fn main() {
    let n = 10; // Size of the problem
    // Generate a test problem
    let (a, b) = generate_test_problem(n);
    // Solve using conjugate gradient method
    println!("Starting conjugate gradient method");
    let start = Instant::now();
    let x = cg::conjugate_grad(&a, &b, None);
    let duration = start.elapsed();
    println!("Conjugate gradient method took: {:?}", duration);
    // Verify the solution
    let residual = cg::verify_solution(&a, &b, &x);
    println!("Residual: {}", residual);
    // Compare with built-in solver
    println!("Starting built-in solver");
    let start = Instant::now();
    let x_builtin = a.lu().solve(&b).unwrap();
    let duration = start.elapsed();
    println!("Built-in solver took: {:?}", duration);
    // Compare solutions
    let diff = (&x - &x_builtin).norm();
    println!("Difference between solutions: {}", diff);
}
fn generate_test_problem(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut rng = rand::thread_rng();
    // Generate a random positive semi-definite matrix
    let p: DMatrix<f64> = DMatrix::from_fn(n, n, |_, _| rng.gen_range(0.0..1.0));
    let a = &p.transpose() * &p;
    let b = DVector::from_element(n, 1.0);
    let n = 6;
    let row_haar_matrix = haar_sample(n, n, sketch::MatrixAttribute::Row);
    println!("Row Orthogonal:\n{}", row_haar_matrix);
    
    let col_haar_matrix = haar_sample(n, n, sketch::MatrixAttribute::Column);
    println!("Column Orthogonal:\n{}", col_haar_matrix);
    
    let gaussian_matrix = sketch::sketching_operator(sketch::DistributionType::Gaussian, n, n);
    println!("Gaussian Matrix:\n{}", gaussian_matrix);
    
    let uniform_matrix = sketch::sketching_operator(sketch::DistributionType::Uniform, n, n);
    println!("Uniform Matrix:\n{}", uniform_matrix);
    
    let rademacher_matrix = sketch::sketching_operator(sketch::DistributionType::Rademacher, n, n);
    println!("Rademacher Matrix:\n{}", rademacher_matrix);
    (a, b)
    
}