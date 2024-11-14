#![allow(dead_code)]
use nalgebra::DMatrix;
use rand::Rng;  
mod errors;
mod sketch;
mod cg;
mod solvers;
mod sketch_and_precondition;
mod sketch_and_solve;
mod pivot_decompositions;
mod cqrrpt;

use crate::cqrrpt::sap_chol_qrcp;
use crate::pivot_decompositions::qrcp;
use crate::sketch::{sketching_operator, DistributionType};
use std::time::Instant;
    
fn permutation_vector_to_transpose_matrix(perm: &[usize]) -> DMatrix<f64> {
    let n = perm.len();
    let mut perm_matrix = DMatrix::<f64>::zeros(n, n); // Initialize an n x n matrix with zeros

    for (i, &p) in perm.iter().enumerate() {
        perm_matrix[(i, p)] = 1.0;
    }

    perm_matrix
}
fn check_approx_equal(a: &DMatrix<f64>, b: &DMatrix<f64>, tolerance: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if (a[(i, j)] - b[(i, j)]).abs() > tolerance {
                println!("{}, {}, {}, {}", i, j, a[(i, j)], b[(i, j)]);
                return false;
            }
        }
    }
    
    true
}

fn check_upper_triangular(a: &DMatrix<f64>, tolerance: f64) -> bool {
    
    for i in 0..a.nrows() {
        for j in 0..i.min(a.ncols()) {
            if (a[(i, j)]).abs() > tolerance {
                // println!("({}, {}), {}", i, j, a[(i, j)]);
                return false;
            }
        }
    }
    true
}
fn main(){
    let n = 100;
    let m = 1000;
    let d = rand::thread_rng().gen_range(n..m);
    
    let data = sketching_operator(DistributionType::Uniform, m, n).unwrap();
    
    let start1 = Instant::now();
    let (q, r, j) = sap_chol_qrcp(&data, d);
    let duration1 = start1.elapsed();

    let start2 = Instant::now();
    let (q_cp, r_cp, p) = qrcp(&data);
    let p_cp = permutation_vector_to_transpose_matrix(&p);
    let duration2 = start2.elapsed();
    

    let reconstruct = &q*&r*permutation_vector_to_transpose_matrix(&j);
    let reconstructed = &q_cp*&r_cp*&p_cp;
    let qtq = &q.transpose()*&q;
    println!("Deterministic Time {:2?}", duration2);
    println!("Sketched Time {:2?}", duration1);
    println!("Reconstruction Error {}", (&reconstruct - &reconstructed).norm());
    
    assert!(check_upper_triangular(&r, 1e-4));
    assert!(check_approx_equal(&qtq,  &DMatrix::identity(q.ncols(), q.ncols()), 1e-4));
    assert!(check_approx_equal(&reconstruct, &reconstructed, 1e-4));
}