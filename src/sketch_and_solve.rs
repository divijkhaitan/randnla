use nalgebra::DMatrix;
use crate::sketch::{sketching_operator, DistributionType};
use crate::solvers::{solve_diagonal_system, solve_upper_triangular_system};

pub fn sketched_least_squares_qr(a:&DMatrix<f64>, b:&DMatrix<f64>) -> DMatrix<f64>
{
    let rows = a.nrows();
    let s: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = sketching_operator(DistributionType::Gaussian, rows/4, rows).unwrap();
    let a_sk = &s*a;
    let mut b_sk = &s*b;
    let (q , r) = a_sk.qr().unpack();
    b_sk = q.transpose()*b_sk;
    let x = solve_upper_triangular_system(&r, &b_sk);
    x
}

pub fn sketched_least_squares_svd(a:&DMatrix<f64>, b:&DMatrix<f64>) -> DMatrix<f64>
{
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
    #[test]
    fn test_least_squares_qr()
    {
        // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
        let mut rng = rand::thread_rng();
        let n = rand::thread_rng().gen_range(100..300);
        let m = rand::thread_rng().gen_range(n..5000);
        let epsilon = 0.01;
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng));
        let mut data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        let y = &data*&hypothesis;
        for i in 0..m {
            let noise_vector = DMatrix::from_fn(n, 1, |_, _| normal.sample(&mut rng));
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
    fn test_least_squares_svd()
    {
        // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
        let mut rng = rand::thread_rng();
        let n = rand::thread_rng().gen_range(100..300);
        let m = rand::thread_rng().gen_range(n..5000);
        let epsilon = 0.01;
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng));
        let mut data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        let y = &data*&hypothesis;
        for i in 0..m {
            let noise_vector = DMatrix::from_fn(n, 1, |_, _| normal.sample(&mut rng));
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