use nalgebra::DMatrix;
use crate::sketch::{sketching_operator, DistributionType};
use crate::cg;
use crate::errors::RandNLAError;
use std::error::Error;

// Blendenpik Least Squares Solver
/**
Implements the Blendenpik algorithm for solving overdetermined linear systems.

* Inputs:
a: An m x n matrix (must be overdetermined, i.e., m >= n).  
b: An m x 1 matrix (the right-hand side of the system).  
epsilon: Convergence tolerance for the Conjugate Gradient Least Squares (CGLS) solver.  
l: Maximum number of iterations for the CGLS solver.  
sampling_factor: A factor controlling the number of rows to sample during the sketching step.

* Output:
An approximate solution x to the least-squares problem argmin_x ||Ax - b||_2.

This algorithm computes the QR decomposition of a sketch of A, uses the 
upper triangular factor of the sketch as a preconditioner and solves the 
system using CGLS. 
*/

pub fn blendenpik_overdetermined(a:&DMatrix<f64>, b:&DMatrix<f64>, epsilon:f64, l:usize, sampling_factor:f64) -> Result<DMatrix<f64>, Box<dyn Error>> {
    let m = a.nrows();
    let n = a.ncols();
    if m < n{
        return Err(Box::new(RandNLAError::NotOverdetermined(
            format!("Need more columns than rows, found {} rows and {} columns", a.nrows(), a.ncols())
        )));
    }
    if sampling_factor < 1.0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Sampling factor must be greater than 1, current input is {}", sampling_factor)
            )))
    }
    if epsilon <= 0.0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Epsilon must be positive, current input is {}", epsilon)
            )))
    }
    if l == 0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Number of iterations must be positive, current input is {}", l)
            )))
    }
    let d = if sampling_factor*(n as f64) > m as f64 {m} else {(sampling_factor*(n as f64)).floor() as usize};
    let s = sketching_operator(DistributionType::Gaussian, d, m).unwrap();
    let a_sk = &s*a;
    let b_sk = &s*b;
    let (q, r) = a_sk.qr().unpack();
    let z_0 = q.transpose()*&b_sk;
    let rinv = r.solve_upper_triangular(&DMatrix::identity(n, n)).unwrap();
    let a_preconditioned = a*&rinv;
    let z = cg::cgls(&a_preconditioned, &b, epsilon, l, Some(z_0));
    Ok(rinv*z)
}

// LSRN Least Squares Solver
/**
Solves overdetermined linear systems using the LSRN algorithm.

* Inputs:
a: An m x n matrix (must be overdetermined, i.e., `m >= n`).  
b: An m x 1 matrix (the right-hand side of the system).  
epsilon: Convergence tolerance for the Conjugate Gradient Least Squares (CGLS) solver.  
l: Maximum number of iterations for the CGLS solver.  
sampling_factor: A factor controlling the number of rows to sample during the sketching step (must be ≥ 1).

* Output:
x: An approximate solution x to the least-squares problem argmin_x ||Ax - b||_2.

* Error Handling:
Returns an error if the input parameters are invalid or the matrix dimensions are incompatible.

This algorithm computes the SVD of a sketch of A, uses the orthogonal factors 
of the sketch as preconditioners and solves the system using CGLS. 
*/

fn lsrn_overdetermined(a: &DMatrix<f64>, b: &DMatrix<f64>, epsilon:f64, l:usize, sampling_factor: f64) -> Result<DMatrix<f64>, Box<dyn Error>> {
    let m = a.nrows();
    let n = a.ncols();
    if m < n{
        return Err(Box::new(RandNLAError::NotOverdetermined(
            format!("Need more columns than rows, found {} rows and {} columns", a.nrows(), a.ncols())
        )));
    }
    if sampling_factor < 1.0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Sampling factor must be greater than 1, current input is {}", sampling_factor)
            )))
    }
    if epsilon <= 0.0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Epsilon must be positive, current input is {}", epsilon)
            )))
    }
    if l == 0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Number of iterations must be positive, current input is {}", l)
            )))
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

//Saddle Point Solver
/**
Preconditions and solves a saddle-point problem of the form argmin_x ||Ax - b||_2^2 + mu||x||_2^2 + 2<c, x> as well as it's dual. It converts any dual or primal saddle point problem into a primal saddle point problem with the last term eliminated, 

* Inputs:
a: m x n matrix (the primary coefficient matrix).  
b: m x 1 matrix (The vector to minimise matrix product from).  
c: n x 1 matrix (Inner product term).  
mu: Regularization factor (must be ≥ 0).  
epsilon: Convergence tolerance for the Conjugate Gradient Least Squares (CGLS) solver.  
l: Maximum number of iterations for the CGLS solver.  
sampling_factor: Factor controlling the number of rows to sample during sketching (must be ≥ 1).  

* Outputs:
x: Solution to the primal saddle-point problem.  
y: Solution to the dual saddle point problem

* Error Handling:
Returns an error if the input parameters are invalid or if the system is underdetermined (`m < n`).

* Notes:
- Would likely be faster with SRFT instead of gaussian sketch

This algorithm computes an SVD of a sketch and modifies the singular values
depending on the regularisation coefficient. Reduces the problem to a saddle 
point with c = 0 and then a regular least squares problem which is solved by
CGLS.
*/

fn sketch_saddle_point_precondition(a: &DMatrix<f64>, b: &DMatrix<f64>, c: &DMatrix<f64>, mu: f64, epsilon: f64, l: usize, sampling_factor: f64) -> Result<(DMatrix<f64>, DMatrix<f64>), Box<dyn Error>> {
    let (m, n) = a.shape();
    if m < n{
        return Err(Box::new(RandNLAError::NotOverdetermined(
            format!("Need more columns than rows, found {} rows and {} columns", a.nrows(), a.ncols())
        )));
    }
    if sampling_factor < 1.0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Sampling factor must be greater than 1, current input is {}", sampling_factor)
            )))
    }
    if epsilon <= 0.0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Epsilon must be positive, current input is {}", epsilon)
            )))
    }
    if l == 0{
        return Err(Box::new(RandNLAError::InvalidParameters(
            format!("Number of iterations must be positive, current input is {}", 0)
            )))
    }
    let d = ((sampling_factor * n as f64).floor() as usize).max(1).min(m);
    let s = sketching_operator(DistributionType::Gaussian, d, m).unwrap();
    
    // Compute A^sk
    let a_sk = &s * a;

    // Compute SVD of A^sk
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
    // Preconditioned matrix (Can be replaced with Efficient Operator for performance)
    let a_precond = a * &m;
    // Compute b_mod
    let mut b_mod =  b.clone();
    if !c.is_empty() {
        let b_shift = &vt * c;
        let temp = if mu > 0.0 {
            let matrix = DMatrix::from_diagonal(&sigma.map(|x| 1.0 / (x.powi(2) + mu).sqrt()));
            &s.transpose()*(&u*(matrix*b_shift))
        } else {
            // let k = sigma.iter().take_while(|&&s| s > 0).count();
            &s.transpose() * (&u * (DMatrix::from_diagonal(&sigma.map(|s| 1.0 / s))*b_shift))
        };
        b_mod -= temp;
    }
    // Compute initial guess z0
    let z0 = &u.transpose() * &(&s * &b_mod);
    
    // Solve the preconditioned system using CGLS
    let z = cg::cgls(&a_precond, &b_mod, epsilon, l, Some(z0));
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
        let mut a = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
    
        // Modified Gram-Schmidt
        for i in 0..n {
            let mut v = a.column(i).clone_owned();
            for j in 0..i {
                let proj = a.column(j).dot(&v);
                v -= proj * a.column(j);
            }
            v /= v.norm();
            a.set_column(i, &v);
        }
    
        // Scale columns to worsen conditioning
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
        let mut rng = rand::thread_rng();
        let epsilon = 0.0001;
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng));
        let data = {
            if ill_conditioning{
                generate_tall_ill_conditioned_matrix(m, n, 1e6)
            }
            else
            {
                sketching_operator(DistributionType::Gaussian, m, n).unwrap()
            }
        };
        let mut y = &data*&hypothesis;
        let noise_vector = DMatrix::from_fn(m, 1, |_, _| normal.sample(&mut rng));
        for i in 0..m {
            y[(i, 0)] += noise_vector[(i, 0)];
        }
        (data, hypothesis, y)
    }
    
    #[test]
    fn test_blendenpik_overdetermined(){
        let n = 10;
        let m = 100;
        let (data, hypothesis, y) = generate_least_squares_problem(m, n, true);
        
        // Blendenpik Test
        let start_randomised = Instant::now();
        // compute using sketched qr
        let x = blendenpik_overdetermined(&data, &y, 0.0001, 1000, 1.5).unwrap();
        let duration_randomised = start_randomised.elapsed();
        // compute using plain qr
        let start_deterministic = Instant::now();
        let (q, r) = data.clone().qr().unpack();
        let b_transformed = q.transpose()*&y;
        let actual_solution = solve_upper_triangular_system(&r, &b_transformed);
        let duration_deterministic = start_deterministic.elapsed();
        
        // let start_iterative = Instant::now();
        // let iterative_solution = cg::cgls(&data, &y, 0.0001, 1000, Some(DMatrix::zeros(n, 1)));
        // let duration_iterative = start_iterative.elapsed();
        
        
        let residual_hypothesis = &data*&hypothesis - &y;
        let residual_sketch = &data*x - &y;
        let residual_actual = &data*actual_solution - &y;
        // let residual_iterative = &data*iterative_solution - &y;
        
        let x = blendenpik_overdetermined(&data, &y, 0.0001, 1000, 0.5);
        assert!(x.is_err());
        
        let x = blendenpik_overdetermined(&data, &y, -0.0001, 1000, 1.5);
        assert!(x.is_err());
        
        let x = blendenpik_overdetermined(&data, &y, 0.0, 1000, 1.5);
        assert!(x.is_err());
        
        let x = blendenpik_overdetermined(&data, &y, 0.0001, 0, 0.5);
        assert!(x.is_err());
        
        println!("Least Squares Blendenpik test");
        // println!("Hypothesis residual: {}, Actual Residual: {}", (residual_hypothesis).norm(), (residual_actual).norm());
        // println!("Sketched residual: {}, Iterative Residual: {}", (residual_sketch).norm(), (residual_iterative).norm());
        println!("Relative Hypothesis error = {}, Relative Actual Error= {}", residual_hypothesis.norm()/y.norm(), (residual_actual).norm()/y.norm());
        println!("Relative Sketched error = {}, Relative Iterative Error= {}", residual_sketch.norm()/y.norm(), 0.0);
        println!("Times: (sketched, actual): {:.2?} {:.2?}", duration_randomised, duration_deterministic);
        // println!("Times: (sketched, actual, iterative): {:.2?} {:.2?} {:.2?}", duration_randomised, duration_deterministic, duration_iterative);
    }
    #[test]
    fn test_saddle_point(){
        let mut rng = rand::thread_rng();
        let n = 10;
        let m = 100;
        let (data, _, y) = generate_least_squares_problem(m, n, true);
        let uniform = Uniform::new(-10.0, 10.0);
        let c = DMatrix::from_fn(n, 1, |_, _| uniform.sample(&mut rng));
        let mu = uniform.sample(&mut rng);
        // Blendenpik Test
        
        
        let start_randomised = Instant::now();
        // compute using sketched algorithm
        let (sketched_solution, _sketched_dual_solution) = sketch_saddle_point_precondition(&data, &y, &c, mu, 0.0001, 1000, 1.5).unwrap();
        let duration_randomised = start_randomised.elapsed();
        
        // compute using SVD
        let start_deterministic = Instant::now();
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
        let duration_deterministic = start_deterministic.elapsed();
        
        // compute using cgls
        let start_iterative: Instant = Instant::now();
        let ata_mu = &data.transpose()*&data + DMatrix::identity(n, n);
        let atb_c = &data.transpose()*&y + &c;
        let iterative_solution = cg::cgls(&ata_mu, &atb_c, 0.0001, 1000, Some(DMatrix::zeros(n, 1)));
        let duration_iterative = start_iterative.elapsed();
        
        let norm_sketch = (&data*&sketched_solution).norm().powf(2.0) + mu*sketched_solution.norm().powf(2.0) + 2.0*(c.columns(0, 1).dot(&sketched_solution));
        let norm_iterative = (&data*&iterative_solution).norm().powf(2.0) + mu*iterative_solution.norm().powf(2.0) + 2.0*(c.columns(0, 1).dot(&iterative_solution));
        let norm_actual = (&data*&actual_solution).norm().powf(2.0) + mu*actual_solution.norm().powf(2.0) + 2.0*(c.columns(0, 1).dot(&actual_solution));
        
        let x = sketch_saddle_point_precondition(&data, &y, &c, mu, 0.0001, 1000, 0.5);
        assert!(x.is_err());
        
        let x = sketch_saddle_point_precondition(&data, &y, &c, mu, -0.0001, 1000, 1.5);
        assert!(x.is_err());
        
        let x = sketch_saddle_point_precondition(&data, &y, &c, mu, 0.0, 1000, 1.5);
        assert!(x.is_err());
        
        let x = sketch_saddle_point_precondition(&data, &y, &c, mu, 0.0001, 0, 1.5);
        assert!(x.is_err());
        

        println!("Least Squares Saddle Point test");
        println!("Actual Residual: {}, Sketched residual: {}, Iterative Residual: {}", norm_actual, norm_sketch, norm_iterative);
        println!("Times: (Actual, Sketched, Iterative): {:.2?} {:.2?} {:.2?}", duration_deterministic, duration_randomised, duration_iterative);
    }
    
    
    #[test]    
    fn test_lsrn_overdetermined(){
        let n = rand::thread_rng().gen_range(10..50);
        let m = rand::thread_rng().gen_range(100..500);
        let (data, hypothesis, y) = generate_least_squares_problem(m, n, true);
        // LSRN
        let start_randomised = Instant::now();
        // compute using lsrn
        let x = lsrn_overdetermined(&data, &y, 0.0001, 1000, 3.0).unwrap();
        let duration_randomised = start_randomised.elapsed();
        
        // compute using SVD
        let start_deterministic = Instant::now();
        let svd_obj = data.clone().svd(true, true);
        let u = svd_obj.u.unwrap();
        let sigma = DMatrix::from_diagonal(&svd_obj.singular_values);
        let v = svd_obj.v_t.unwrap().transpose();
        let b_transformed = u.transpose()*&y;
        let actual_solution = v*solve_diagonal_system(&sigma, &b_transformed);
        let duration_deterministic = start_deterministic.elapsed();

        // compute using iterative solver
        let start_iterative = Instant::now();
        let iterative_solution = cg::cgls(&data, &y, 0.0001, 1000, Some(DMatrix::zeros(n, 1)));
        let duration_iterative = start_iterative.elapsed();
        
        let residual_hypothesis = &data*&hypothesis - &y;
        let residual_sketch = &data*x - &y;
        let residual_actual = &data*actual_solution - &y;
        let residual_iterative = &data*iterative_solution - &y;
        
        let x = lsrn_overdetermined(&data, &y, 0.0001, 1000, 0.5);
        assert!(x.is_err());
        
        let x = lsrn_overdetermined(&data, &y, -0.0001, 1000, 1.5);
        assert!(x.is_err());
        
        let x = lsrn_overdetermined(&data, &y, 0.0, 1000, 1.5);
        assert!(x.is_err());
        
        let x = lsrn_overdetermined(&data, &y, 0.0001, 0, 0.5);
        assert!(x.is_err());
        
        println!("Least Squares LSRN test");
        println!("Hypothesis residual: {}, Actual Residual: {}", (residual_hypothesis).norm(), (residual_actual).norm());
        println!("Sketched residual: {}, Iterative Residual: {}", (residual_sketch).norm(), (residual_iterative).norm());
        println!("Relative Hypothesis error = {}, Relative Actual Error= {}", residual_hypothesis.norm()/y.norm(), (residual_actual).norm()/y.norm());
        println!("Relative Sketched error = {}, Relative Iterative Error= {}", residual_sketch.norm()/y.norm(), (residual_iterative).norm()/y.norm());
        println!("Times: (sketched, actual, iterative): {:.2?} {:.2?} {:.2?}", duration_randomised, duration_deterministic, duration_iterative);
    }
    
}