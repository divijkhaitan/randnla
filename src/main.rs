mod sample;
mod cg;

use nalgebra::{DMatrix, DVector, dmatrix, dvector};
use rand::Rng;
use std::time::Instant;

fn main() {
    test_solvers();
   
}










// Testing Solvers ========================================================================
fn test_solvers() {
    let n = 3; 
    let (a, b) = generate_test_problem(n);
    let a = dmatrix![4.0, 1.0; 1.0, 3.0];
    let b = dvector![1.0,2.0];
    



    println!("Testing with n = {}", n);
    println!("Matrix A: \n{}", a);
    println!("Vector b: \n{}", b);
    test_cg_method(a, b);
} 
fn generate_test_problem(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut rng = rand::thread_rng();
    
    let p: DMatrix<f64> = DMatrix::from_fn(n, n, |_, _| rng.gen_range(0.0..1.0));
    let a = &p.transpose() * &p;
    let b = DVector::from_element(n, 1.0);
    (a, b)
}

fn test_cg_method(a: DMatrix<f64>, b: DVector<f64>) {
    println!("Starting conjugate gradient method");
    let start = Instant::now();
    let x = cg::conjugate_grad(&a, &b, None);
    println!("Solution x : {}\n", x);
    let duration = start.elapsed();
    println!("Conjugate gradient method took: {:?}", duration);
    
    let residual = cg::verify_solution(&a, &b, &x);
    println!("Residual: {}", residual);
    
    println!("Starting built-in solver");
    let start = Instant::now();
    let x_builtin = a.lu().solve(&b).unwrap();
    let duration = start.elapsed();
    println!("Built-in solver took: {:?}", duration);
    
    let diff = (&x - &x_builtin).norm();
    println!("Difference between solutions: {}", diff);
}
// Testing Solvers ========================================================================