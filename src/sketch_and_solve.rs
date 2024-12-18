use nalgebra::DMatrix;
use crate::sketch::{sketching_operator, DistributionType};
use crate::solvers::{solve_diagonal_system, solve_upper_triangular_system};

// Sketch and Solve using QR factorisation  
/**
Implements sketched least squares solvers using QR decomposition.  

* Inputs:  
a: m x n matrix (the primary coefficient matrix)  
b: m x 1 matrix (the right-hand side vector)  

* Output:  
x: An approximate solution x to the least-squares problem argmin_x ||Ax - b||_2  

* Notes:  
The original Blendenpik uses a fast subsampled trignometric transform, which may offer a boost in performance  
Better suited to well conditioned problems, use SVD sketch-and-solve if unknown  

Computes the QR factorisations of the sketch and then uses it to solve the system  
*/


pub fn sketched_least_squares_qr(a:&DMatrix<f64>, b:&DMatrix<f64>) -> DMatrix<f64>{
    let rows = a.nrows();
    let s: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = sketching_operator(DistributionType::Gaussian, rows/4, rows).unwrap();
    let a_sk = &s*a;
    let mut b_sk = &s*b;
    let (q , r) = a_sk.qr().unpack();
    b_sk = q.transpose()*b_sk;
    let x = solve_upper_triangular_system(&r, &b_sk);
    x
}

// Sketch and Solve using Singular Value Decomposition  
/**
Implements sketched least squares solvers using SVD.  

* Inputs:  
a: m x n matrix (the primary coefficient matrix)    
b: m x 1 matrix (the right-hand side vector)  

* Output:  
x: An approximate solution x to the least-squares problem argmin_x ||Ax - b||_2  

* Notes:  
Robust to ill conditioned problems  
Panics when SVD fails  
Computes the SVD of the sketch and then uses it to solve the system  

*/

pub fn sketched_least_squares_svd(a:&DMatrix<f64>, b:&DMatrix<f64>) -> DMatrix<f64>{
    let rows = a.nrows();
    let sketchop = sketching_operator(DistributionType::Gaussian, rows/4, rows).unwrap();
    let a_sk = &sketchop*a;
    let mut b_sk = &sketchop*b;
    let svd_obj = a_sk.svd(true, true);
    let u = svd_obj.u.unwrap();
    let sigma = DMatrix::from_diagonal(&svd_obj.singular_values);
    let v = svd_obj.v_t.unwrap().transpose();
    b_sk = u.transpose()*b_sk;
    let x = solve_diagonal_system(&sigma, &b_sk);
    v*x
}



#[cfg(test)]
mod tests
{
    use std::time::Instant;
    use rand::Rng;
    use rand_distr::{Uniform, Normal, Distribution};
    use nalgebra::DMatrix;
    use super::{sketched_least_squares_qr, sketched_least_squares_svd};
    use crate::{sketch::{sketching_operator, DistributionType}, solvers::{solve_upper_triangular_system, solve_diagonal_system}};
    use rand_123::rng::ThreeFry2x64Rng;
    use rand_core::SeedableRng;
    #[test]
    fn test_least_squares_qr(){
        // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
        let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
        let n = rng_threefry.gen_range(10..30);
        let m = rng_threefry.gen_range(n..500);
        let epsilon = 0.01;
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng_threefry));
        let mut data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        let y = &data*&hypothesis;
        for i in 0..m {
            let noise_vector = DMatrix::from_fn(n, 1, |_, _| normal.sample(&mut rng_threefry));
            for j in 0..n {
                data[(i, j)] += noise_vector[(j, 0)];
            }
        }
        // compute using sketched qr
        let start1 = Instant::now();
        let x = sketched_least_squares_qr(&data, &y);
        let duration1 = start1.elapsed();
        
        // compute using plain qr
        let start2 = Instant::now();
        let (q, r) = data.clone().qr().unpack();
        let b_transformed = q.transpose()*&y;
        let actual_solution = solve_upper_triangular_system(&r, &b_transformed);
        let duration2 = start2.elapsed();
        
        let residual_hypothesis = &data*&hypothesis - &y;
        let residual_actual = &data*&actual_solution - &y;
        let residual_sketch = &data*&x - &y;
        println!("Least Squares QR test");
        println!("Hypothesis residual: {}, Actual Residual: {}, Sketched residual: {}", (residual_hypothesis).norm(), (residual_actual).norm(), (residual_sketch).norm());
        println!("Relative Hypothesis error = {}, Relative Actual Error= {}, Relative Sketched Error: {}", residual_hypothesis.norm()/y.norm(), (residual_actual).norm()/y.norm(), residual_sketch.norm()/y.norm());
        println!("Times: (sketched, actual): {:.2?}, {:.2?}", duration1, duration2);
    }
    #[test]
    fn test_least_squares_svd(){
        // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
        let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
        let n = rng_threefry.gen_range(10..30);
        let m = rng_threefry.gen_range(n..500);
        let epsilon = 0.01;
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng_threefry));
        let mut data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        let y = &data*&hypothesis;
        for i in 0..m {
            let noise_vector = DMatrix::from_fn(n, 1, |_, _| normal.sample(&mut rng_threefry));
            for j in 0..n {
                data[(i, j)] += noise_vector[(j, 0)];
            }
        }
        // compute using sketched SVD
        let start1 = Instant::now();
        let x = sketched_least_squares_svd(&data, &y);
        let duration1 = start1.elapsed();
        
        // compute using SVD
        let start2 = Instant::now();
        let svd_obj = data.clone().svd(true, true);
        let u = svd_obj.u.unwrap();
        let sigma = DMatrix::from_diagonal(&svd_obj.singular_values);
        let v = svd_obj.v_t.unwrap().transpose();
        let b_transformed = u.transpose()*&y;
        let actual_solution = v*solve_diagonal_system(&sigma, &b_transformed);
        let duration2 = start2.elapsed();
        
        let residual_hypothesis = &data*&hypothesis - &y;
        let residual_actual = &data*&actual_solution - &y;
        let residual_sketch = &data*&x - &y;
        
        println!("Least Squares SVD test");
        println!("Hypothesis residual: {}, Actual Residual: {}, Sketched residual: {}", (residual_hypothesis).norm(), (residual_actual).norm(), (residual_sketch).norm());
        println!("Relative Hypothesis error = {}, Relative Actual Error= {}, Relative Sketched Error: {}", residual_hypothesis.norm()/y.norm(), (residual_actual).norm()/y.norm(), residual_sketch.norm()/y.norm());
        println!("Times: (sketched, actual): {:.2?}, {:.2?}", duration1, duration2);
    }
}