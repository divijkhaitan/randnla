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
use std::time::{Instant, Duration};
use std::fs::File;
use std::io::Write;

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
fn main() {
    let mut csv_data = String::new();
    csv_data.push_str("Matrix Size,Sketched Time,Deterministic Time,Reconstruction Error\n");

    let sizes = vec![500, 1000, 1500];

    for &m in &sizes {
        for _ in 0..5{
            let n = m / 10;
            let d = rand::thread_rng().gen_range(n..m);

            let data = sketching_operator(DistributionType::Uniform, m, n).unwrap();
            let data_clone = data.clone();

            let start1 = Instant::now();
            let (q, r, j) = match run_with_timeout(move || sap_chol_qrcp(&data, d), Duration::from_secs(300)) {
                Ok((q, r, j)) => (q, r, j),
                Err(_) => {
                    csv_data.push_str(&format!("{},{:?},Timeout,Timeout,Timeout,Timeout\n", m, d));
                    continue;
                }
            };
            let duration1 = start1.elapsed();

            let start2 = Instant::now();
            let (q_cp, r_cp, p) = match run_with_timeout(move || qrcp(&data_clone), Duration::from_secs(300)) {
                Ok((q_cp, r_cp, p)) => (q_cp, r_cp, p),
                Err(_) => {
                    csv_data.push_str(&format!("{},{:?},{:?},Timeout,Timeout,Timeout\n", m, d, duration1));
                    continue;
                }
            };
            let duration2 = start2.elapsed();

            let reconstruct = &q * &r * permutation_vector_to_transpose_matrix(&j);
            let reconstructed = &q_cp*&r_cp*permutation_vector_to_transpose_matrix(&p);
            // p.inv_permute_columns(&mut reconstructed);
            let reconstruction_error = (&reconstruct - &reconstructed).norm();

            csv_data.push_str(&format!(
                "{},{:?},{:?},{}\n",
                m, duration1, duration2, reconstruction_error
            ));

            assert!(check_upper_triangular(&r, 1e-4));
            assert!(check_approx_equal(
                &(&q.transpose() * &q),
                &DMatrix::identity(q.ncols(), q.ncols()),
                1e-4
            ));
            assert!(check_approx_equal(&reconstruct, &reconstructed, 1e-4));
        }
    }

    let mut file = File::create("cqrrpt_results_cpqr.csv").expect("Unable to create file");
    file.write_all(csv_data.as_bytes())
        .expect("Unable to write data to file");
}

fn run_with_timeout<T, F>(f: F, timeout: Duration) -> Result<T, ()>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let result = f();
        tx.send(result).unwrap();
    });

    match rx.recv_timeout(timeout) {
        Ok(result) => Ok(result),
        Err(_) => Err(()),
    }
}