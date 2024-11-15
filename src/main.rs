#![allow(warnings)]
use nalgebra::{DMatrix, DVector, dmatrix, dvector};
use rand_core::{SeedableRng, RngCore};
use rand::Rng;
use std::time::Instant;

use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};

use rand::distributions::DistIter;

use rand_123::rng::ThreeFry2x64Rng;

mod sample;
mod sketch;
mod cg;
mod solvers;
mod sketch_and_precondition;
mod sketch_and_solve;
mod lora_drivers;
mod lora_helpers;



pub enum DistributionType {
    Gaussian,
    Uniform,
    Rademacher,
}


fn main() {

    let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
    
}













fn test_drivers(){
    let a = dmatrix![1.0, 2.0, 3.0;
                 4.0, 5.0, 6.0;
                 7.0, 8.0, 9.0];

    let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let dims = 1000;
    let a =  DMatrix::from_fn(dims, dims, |_i, _j| normal.sample(&mut rng_threefry));
    // println!("A: \n{}", a);

    let k = dims;
    let epsilon= 0.01;
    let s = 5;

    let tick = Instant::now();
    let (u, s, v) = lora_drivers::rand_SVD(&a, k, epsilon, s);
    let tock = tick.elapsed();

    println!("Time taken by RandSVD: {:?}", tock);
    // println!("U: \n{}", u);
    // println!("S: \n{}", s);
    // println!("V: \n{}", v);

    // normal nalgebra svd

    let tick = Instant::now();
    let svd = a.svd(true, true);
    let tock = tick.elapsed();
    println!("Time taken by Deterministic SVD: {:?}", tock);


    let deterministic_u = svd.u.unwrap();
    let deterministic_s = DMatrix::from_diagonal(&svd.singular_values);
    let deterministic_v = svd.v_t.unwrap().transpose();
    // println!("Deterministic U: \n{}", deterministic_u);
    // println!("Deterministic S: \n{}", deterministic_s);
    // println!("Deterministic V: \n{}", deterministic_v);

    let diff_u = (&u - &deterministic_u).norm();
    let diff_s = (&s - &deterministic_s).norm();
    let diff_v = (&v - &deterministic_v).norm();
    println!("Difference between U: {}", diff_u);
    println!("Difference between S: {}", diff_s);
    println!("Difference between V: {}", diff_v);





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