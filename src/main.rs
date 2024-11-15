#![allow(dead_code)]
#![allow(unused_imports)]
use nalgebra::{DMatrix, DVector, dmatrix, dvector};
use rand_core::{SeedableRng, RngCore};
use rand::Rng;
use std::time::Instant;

use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};

use rand::distributions::DistIter;

use rand_123::rng::ThreeFry2x64Rng;

mod errors;
mod sketch;
mod cg;
mod solvers;
mod sketch_and_precondition;
mod sketch_and_solve;



pub enum DistributionType {
    Gaussian,
    Uniform,
    Rademacher,
}


fn main() {


    let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
    let threefry_normal: DistIter<StandardNormal, &mut ThreeFry2x64Rng, f64> = StandardNormal.sample_iter(&mut rng_threefry);
    let data: Vec<f64> = threefry_normal.take(2 * 2).collect();

    println!("Threefry Data: {:?}", data);

    // test_solvers();
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
    let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
    
    
    let p: DMatrix<f64> = DMatrix::from_fn(n, n, |_, _| rng_threefry.gen_range(0.0..1.0));
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