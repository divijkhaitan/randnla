use nalgebra::DMatrix;
use crate::sketch::{sketching_operator, DistributionType};
use crate::cg;
use rand::distributions::DistIter;
use rand_123::rng::ThreeFry2x64Rng;
use rand_core::SeedableRng;
use crate::errors::DimensionError;
use std::error::Error;

pub fn blendenpik_overdetermined(a:&DMatrix<f64>, b:&DMatrix<f64>, epsilon:f64, l:usize, sampling_factor:f64) -> Result<DMatrix<f64>, Box<dyn Error>> {
    let m = a.nrows();
    let n = a.ncols();
    if m < n{
        return Err(Box::new(DimensionError::NotOverdetermined(
            format!("Need more columns than rows, found {} rows and {} columns", a.nrows(), a.ncols())
        )));
    }
    let d = if sampling_factor*(n as f64) > m as f64 {m} else {(sampling_factor*(n as f64)).floor() as usize};
    let s = sketching_operator(DistributionType::Gaussian, d, m).unwrap();
    let a_sk = &s*a;
    let b_sk = &s*b;
    let (q, r) = a_sk.qr().unpack();
    println!("b shape :({} {}), q shape: ({} {})",b_sk.nrows(), b_sk.ncols(), q.nrows(), q.ncols());
    let z_0 = q.transpose()*&b_sk;
    let rinv = r.try_inverse().unwrap();
    let a_preconditioned = a*&rinv;
    let z = cg::cgls(&a_preconditioned, &b, epsilon, l, Some(z_0));
    Ok(rinv*z)
}

fn lsrn_overdetermined(a: &DMatrix<f64>, b: &DMatrix<f64>, epsilon:f64, l:usize, sampling_factor: f64) -> Result<DMatrix<f64>, Box<dyn Error>> {
    let m = a.nrows();
    let n = a.ncols();
    if m < n{
        return Err(Box::new(DimensionError::NotOverdetermined(
            format!("Need more columns than rows, found {} rows and {} columns", a.nrows(), a.ncols())
        )));
    }
    let d = if sampling_factor*(n as f64) > m as f64 {m} else {(sampling_factor*(n as f64)).floor() as usize};
    let s = sketching_operator(DistributionType::Gaussian, d, m).unwrap();
    let a_sk = &s * a;

    let svd_obj = a_sk.svd(false, true);
    let sigma = DMatrix::from_diagonal(&svd_obj.singular_values);
    let v = svd_obj.v_t.unwrap().transpose();
    
    let sigma_inv = sigma.map(|x| if x != 0.0 { 1.0 / x } else { 0.0 });
    let n = v*&sigma_inv;
    let a_precond = a*&n;
    let mut y_hat = DMatrix::zeros(sigma_inv.ncols(), 1);
    y_hat = cg::cgls(&a_precond, b, epsilon, l,  Some(y_hat), );
    Ok(n* y_hat)
}

fn sketch_saddle_point_precondition(a: &DMatrix<f64>, b: &DMatrix<f64>, c: &DMatrix<f64>, mu: f64, epsilon: f64, l: usize, sampling_factor: f64) -> Result<(DMatrix<f64>, DMatrix<f64>), Box<dyn Error>> {
    let (m, n) = a.shape();
    if m < n{
        return Err(Box::new(DimensionError::NotOverdetermined(
            format!("Need more columns than rows, found {} rows and {} columns", a.nrows(), a.ncols())
        )));
    }
    let d = ((sampling_factor * n as f64).floor() as usize).max(1).min(m);
    let s = sketching_operator(DistributionType::Gaussian, d, m).unwrap();
    
    // Compute A^sk
    let a_sk = &s * a;

    // Compute SVD of A^sk
    let (a_sk_nrows, a_sk_ncols) = a_sk.shape();
    let svd_obj = a_sk.svd(true, true);
    let u = svd_obj.u.unwrap();
    let sigma = svd_obj.singular_values;
    let vt = svd_obj.v_t.unwrap();
        
    // Construct preconditioner M
    let m = if mu > 0.0 {
        &vt.transpose() * DMatrix::from_diagonal(&sigma.map(|s| 1.0 / (s.powi(2) + mu).sqrt()))
    } else {
        let k = sigma.iter().take_while(|&&s| s > 1e-10).count();
        &vt.transpose().columns(0, k) * DMatrix::from_diagonal(&sigma.rows(0, k).map(|s| 1.0 / s))
    };
    println!("a shape ({}, {}), b shape ({}, {}), c shape ({}, {})", a.nrows(), a.ncols(), b.nrows(), b.ncols(), c.nrows(), c.ncols());
    println!("a_sk shape ({}, {}), m shape ({}, {})", a_sk_nrows, a_sk_ncols, m.nrows(), m.ncols());
    println!("u shape ({}, {}), sigma shape ({}, {}), v shape ({}, {})", u.nrows(), u.ncols(), sigma.nrows(), sigma.ncols(), vt.nrows(), vt.ncols());
    
    // Preconditioned matrix (Can be replaced with Efficient Operator for performance)
    let a_precond = a * &m;
    println!("a_precond shape ({}, {}), s shape ({}, {}), b shape ({}, {})", a_precond.nrows(), a_precond.ncols(), s.nrows(), s.ncols(), b.nrows(), b.ncols());

    // Compute b_mod
    let mut b_mod =  b.clone();
    if !c.is_empty() {
        let b_shift = &vt * c;
        println!("b_shift Shape: ({}, {})", b_shift.nrows(), b_shift.ncols());
        let temp = if mu > 0.0 {
            println!("Mu Nonzero");
            let matrix = DMatrix::from_diagonal(&sigma.map(|x| 1.0 / (x.powi(2) + mu).sqrt()));
            &s.transpose()*(&u*(matrix*b_shift))
        } else {
            // let k = sigma.iter().take_while(|&&s| s > 0).count();
            &s.transpose() * (&u * (DMatrix::from_diagonal(&sigma.map(|s| 1.0 / s))*b_shift))
        };
        println!("Temp Shape ({}, {}), b_mod Shape ({}, {})", temp.nrows(), temp.ncols(), b_mod.nrows(), b_mod.ncols());
        b_mod -= temp;
    }
    // println!("a shape ({}, {}), b shape ({}, {}), c shape ({}, {})", a.nrows(), a.ncols(), b.nrows(), b.ncols(), c.nrows(), c.ncols());
    // Compute initial guess z0
    let z0 = &u.transpose() * &(&s * &b_mod);
    println!("Guessed");

    // Solve the preconditioned system using CGLS
    println!("a shape ({}, {}), b shape ({}, {}), c shape ({}, {}), s shape ({}, {})", a.nrows(), a.ncols(), b.nrows(), b.ncols(), c.nrows(), c.ncols(), s.nrows(), s.ncols());
    println!("u shape ({}, {}), sigma shape ({}, {}), v shape ({}, {}), c shape ({}, {})", u.nrows(), u.ncols(), sigma.nrows(), sigma.ncols(), vt.nrows(), vt.ncols(), c.nrows(), c.ncols());
    println!("a shape ({}, {}), b shape ({}, {}), c shape ({}, {}), s shape ({}, {})", a_precond.nrows(), a_precond.ncols(), b_mod.nrows(), b_mod.ncols(), c.nrows(), c.ncols(), s.nrows(), s.ncols());
    let z = cg::cgls(&a_precond, &b_mod, epsilon, l, Some(z0));
    println!("Solved");

    // Compute final solution
    let x = &m * &z;
    let y = b - a * &x;

    Ok((x, y))
}


#[cfg(test)]
mod tests
{
    use rand::Rng;
    use rand_distr::{Uniform, Normal, Distribution};
    use nalgebra::{DMatrix, DVector};
    use crate::{sketch::{sketching_operator, DistributionType}, sketch_and_precondition::{blendenpik_overdetermined, lsrn_overdetermined, sketch_saddle_point_precondition}, solvers::{solve_diagonal_system, solve_upper_triangular_system}};
    use std::time::Instant;
    use crate::cg;
    
    fn generate_tall_ill_conditioned_matrix(m: usize, n: usize, condition_number: f64) -> DMatrix<f64> {
        assert!(m > n, "m must be greater than n for a tall matrix");
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
    
        // Generate a random tall matrix
        let mut a = DMatrix::from_fn(m, n, |_, _| normal.sample(&mut rng));
    
        // Perform a modified Gram-Schmidt process
        for i in 0..n {
            let mut v = a.column(i).clone_owned();
            for j in 0..i {
                let proj = a.column(j).dot(&v);
                v -= proj * a.column(j);
            }
            v /= v.norm();
            a.set_column(i, &v);
        }
    
        // Scale the columns to achieve the desired condition number
        let singular_values = DVector::from_fn(n, |i, _| {
            if i == 0 {
                condition_number
            } else if i == n - 1 {
                1.0
            } else {
                rng.gen_range(1.0..condition_number)
            }
        });
    
        for i in 0..n {
            let mut col = a.column_mut(i);
            col *= singular_values[i];
        }
    
        a
    }
    
    fn generate_least_squares_problem(m:usize , n:usize, ill_conditioning:bool)  -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
        let start = Instant::now();
        let mut rng = rand::thread_rng();
        let epsilon = 0.0001;
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng));
        println!("Time for Generation :{:.2?}", start.elapsed());
        let data = {
            if ill_conditioning{
                generate_tall_ill_conditioned_matrix(m, n, 1e5)
            }
            else
            {
                sketching_operator(DistributionType::Gaussian, m, n).unwrap()
            }
        };
        println!("Time for Generation :{:.2?}", start.elapsed());
        let mut y = &data*&hypothesis;
        let noise_vector = DMatrix::from_fn(m, 1, |_, _| normal.sample(&mut rng));
        for i in 0..m {
            y[(i, 0)] += noise_vector[(i, 0)];
        }
        (data, hypothesis, y)
    }
    
    #[test]
    fn test_blendenpik_overdetermined(){
        let n = rand::thread_rng().gen_range(100..150);
        let m = rand::thread_rng().gen_range(500..5000);
        let (data, hypothesis, y) = generate_least_squares_problem(m, n, true);
        
        // Blendenpik Test
        let start1 = Instant::now();
        // compute using sketched qr
        let x = blendenpik_overdetermined(&data, &y, 0.0001, 10000, 1.5).unwrap();
        let duration1 = start1.elapsed();
        // compute using plain qr
        let start2 = Instant::now();
        let (q, r) = data.clone().qr().unpack();
        let b_transformed = q.transpose()*&y;
        let actual_solution = solve_upper_triangular_system(&r, &b_transformed);
        let duration2 = start2.elapsed();
        

        let start3 = Instant::now();
        let iterative_solution = cg::cgls(&data, &y, 0.0001, 10000, Some(DMatrix::zeros(n, 1)));
        let duration3 = start3.elapsed();
        
        
        let residual_hypothesis = &data*&hypothesis - &y;
        let residual_sketch = &data*x - &y;
        let residual_actual = &data*actual_solution - &y;
        let residual_iterative = &data*iterative_solution - &y;
        
        println!("Hypothesis residual: {}, Actual Residual: {}", (residual_hypothesis).norm(), (residual_actual).norm());
        println!("Sketched residual: {}, Iterative Residual: {}", (residual_sketch).norm(), (residual_iterative).norm());
        println!("Relative Hypothesis error = {}, Relative Actual Error= {}", residual_hypothesis.norm()/y.norm(), (residual_actual).norm()/y.norm());
        println!("Relative Sketched error = {}, Relative Iterative Error= {}", residual_sketch.norm()/y.norm(), (residual_iterative).norm()/y.norm());
        println!("Times: (sketched, actual, iterative): {:.2?} {:.2?} {:.2?}", duration1, duration2, duration3);
    }
    #[test]
    fn test_saddle_point(){
        let mut rng = rand::thread_rng();
        let n = rand::thread_rng().gen_range(100..150);
        let m = rand::thread_rng().gen_range(500..5000);
        let (data, _, y) = generate_least_squares_problem(m, n, true);
        let uniform = Uniform::new(-10.0, 10.0);
        let c = DMatrix::from_fn(n, 1, |_, _| uniform.sample(&mut rng));
        let mu = uniform.sample(&mut rng);
        // Blendenpik Test
        
        
        let start1 = Instant::now();
        // compute using sketched algorithm
        println!("data shape ({}, {}), y shape ({}, {}), c shape ({}, {})", data.nrows(), data.ncols(), y.nrows(), y.ncols(), c.nrows(), c.ncols());
        let (sketched_solution, _) = sketch_saddle_point_precondition(&data, &y, &c, mu, 0.0001, 10000, 1.5).unwrap();
        let duration1 = start1.elapsed();
        
        // compute using SVD
        let start2 = Instant::now();
        // let ata_mu = &data.transpose()*&data + DMatrix::identity(n, n);
        let atb_c = &data.transpose()*&y + &c;
        let svd_obj = data.clone().svd(false, true);
        // let sigma = DMatrix::from_diagonal(&svd_obj.singular_values);
        let vt = svd_obj.v_t.unwrap();
        let pseudoinverse = DVector::from_iterator(
            svd_obj.singular_values.len(),
            svd_obj.singular_values.iter().map(|&s| 1.0 / (mu + s.powi(2)))
        );
        let vt_atb_c = &vt * &atb_c;
        let scaled_vt_atb_c = vt_atb_c.component_mul(&pseudoinverse);
        let actual_solution = &vt.transpose() * &scaled_vt_atb_c;
        let duration2 = start2.elapsed();
        
        // compute using cgls
        let start3 = Instant::now();
        let ata_mu = &data.transpose()*&data + DMatrix::identity(n, n);
        let atb_c = &data.transpose()*&y + &c;
        let iterative_solution = cg::cgls(&ata_mu, &atb_c, 0.0001, 10000, Some(DMatrix::zeros(n, 1)));
        let duration3 = start3.elapsed();
        
        
        let norm_sketch = (&data*&sketched_solution).norm().powf(2.0) + mu*sketched_solution.norm().powf(2.0) + 2.0*(c.columns(0, 1).dot(&sketched_solution));
        let norm_iterative = (&data*&iterative_solution).norm().powf(2.0) + mu*iterative_solution.norm().powf(2.0) + 2.0*(c.columns(0, 1).dot(&iterative_solution));
        let norm_actual = (&data*&actual_solution).norm().powf(2.0) + mu*actual_solution.norm().powf(2.0) + 2.0*(c.columns(0, 1).dot(&actual_solution));
        
        println!("Actual Residual: {}, Sketched residual: {}, Iterative Residual: {}", norm_actual, norm_sketch, norm_iterative);
        println!("Times: (sketched, actual, iterative): {:.2?} {:.2?} {:.2?}", duration1, duration2, duration3);
    }
    
    
    #[test]    
    fn test_lsrn_overdetermined(){
        let start = Instant::now();
        let n = rand::thread_rng().gen_range(100..150);
        let m = rand::thread_rng().gen_range(500..5000);
        let (data, hypothesis, y) = generate_least_squares_problem(m, n, true);
        println!("Time to generate data :{:.2?}", start.elapsed());
        // LSRN
        let start1 = Instant::now();
        // compute using lsrn
        let x = lsrn_overdetermined(&data, &y, 0.0001, 10000, 3.0).unwrap();
        let duration1 = start1.elapsed();
        println!("Time for LSRN :{:.2?}", start.elapsed());
        
        // compute using SVD
        let start2 = Instant::now();
        let svd_obj = data.clone().svd(true, true);
        let u = svd_obj.u.unwrap();
        let sigma = DMatrix::from_diagonal(&svd_obj.singular_values);
        let v = svd_obj.v_t.unwrap().transpose();
        let b_transformed = u.transpose()*&y;
        let actual_solution = v*solve_diagonal_system(&sigma, &b_transformed);
        let duration2 = start2.elapsed();
        println!("Time for SVD :{:.2?}", start.elapsed());

        // compute using iterative solver
        let start3 = Instant::now();
        let iterative_solution = cg::cgls(&data, &y, 0.0001, 10000, Some(DMatrix::zeros(n, 1)));
        let duration3 = start3.elapsed();
        println!("Time for CGLS :{:.2?}", start.elapsed());
        
        let residual_hypothesis = &data*&hypothesis - &y;
        let residual_sketch = &data*x - &y;
        let residual_actual = &data*actual_solution - &y;
        let residual_iterative = &data*iterative_solution - &y;
        
        println!("Hypothesis residual: {}, Actual Residual: {}", (residual_hypothesis).norm(), (residual_actual).norm());
        println!("Sketched residual: {}, Iterative Residual: {}", (residual_sketch).norm(), (residual_iterative).norm());
        println!("Relative Hypothesis error = {}, Relative Actual Error= {}", residual_hypothesis.norm()/y.norm(), (residual_actual).norm()/y.norm());
        println!("Relative Sketched error = {}, Relative Iterative Error= {}", residual_sketch.norm()/y.norm(), (residual_iterative).norm()/y.norm());
        println!("Times: (sketched, actual, iterative): {:.2?} {:.2?} {:.2?}", duration1, duration2, duration3);
    }
    
}