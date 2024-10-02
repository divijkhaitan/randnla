use nalgebra::DMatrix;
use crate::sketch::{sketching_operator, DistributionType};
// use crate::sketch_and_solve::sketched_least_squares_qr;
// use crate::solvers::{solve_diagonal_system, solve_upper_triangular_system};

fn compute_vector_norm(x:&DMatrix<f64>) -> f64{
    let mut norm = 0.0;
    if x.nrows() == 1{
        for i in 0..x.ncols(){
            norm+= (x[(i, 0)]).powf(2.0);
        }
    }
    else if x.ncols() == 1{
        for i in 0..x.nrows(){
            norm+= (x[(i, 0)]).powf(2.0);
        }
    }
    else{
        println!("Not a vector");
    }
    norm
}

fn compute_vector_norm_in_matrix(y:&DMatrix<f64>, a:&DMatrix<f64>) -> f64{
    let x = y.transpose()*a*y;
    let mut norm = 0.0;
    if x.nrows() == 1{
        for i in 0..x.ncols(){
            norm+= (x[(0, i)]).powf(2.0);
        }
    }
    else if x.ncols() == 1{
        for i in 0..x.nrows(){
            norm+= (x[(0, i)]).powf(2.0);
        }
    }
    else{
        println!("Not a vector");
    }
    norm
}

pub fn cgls(a: &DMatrix<f64>, b: &DMatrix<f64>, epsilon: f64, l: usize, z_0: &DMatrix<f64>) -> DMatrix<f64> {
    
    let a_t = a.transpose();  // Transpose of A
    let refat = &a_t;
    let a_t_a = refat*a;
    let b_t = refat * b;  // A^T * b
    let mut r = b_t - &a_t_a * z_0;  // Residual: A^T * (b - A * z_0)
    
    if compute_vector_norm(&r) < epsilon {
        return z_0.clone();
    }
    
    let mut p = r.clone();
    let mut z = z_0.clone();  // Current solution estimate
    let mut z_act: DMatrix<f64> = DMatrix::zeros(a.ncols(), 1);  // Placeholder for the active solution

    for _ in 0..l {
        // Compute alpha_k = (r^T * r) / (p^T * (A^T * A) * p)
        let alpha_k = (compute_vector_norm(&r).powi(2)) / (compute_vector_norm_in_matrix(&p, &(refat * a)));
        let alpha_scaled = alpha_k * &p;

        // Update the solution z = z + alpha_k * p
        z_act = z.clone() + alpha_scaled;
        z = z_act.clone();

        // Compute new residual r_new = r - alpha_k * (A^T * A) * p
        let r_new = &r - alpha_k * &a_t_a * &p;

        // Check for convergence
        if compute_vector_norm(&r_new) < epsilon {
            break;
        }

        // Compute beta_k = (r_new^T * r_new) / (r^T * r)
        let beta_k = compute_vector_norm(&r_new).powi(2) / compute_vector_norm(&r).powi(2);

        // Update p = r_new + beta_k * p
        p = r_new.clone() + beta_k * &p;
        r = r_new;
    }

    z.clone()
}

pub fn blendenpik_least_squares_overdetermined(a:&DMatrix<f64>, b:&DMatrix<f64>, epsilon:f64, l:usize, sampling_factor:f64) -> DMatrix<f64>
{
    let d = if sampling_factor*(a.ncols() as f64) > a.nrows() as f64 {a.nrows()} else {(sampling_factor*(a.ncols() as f64)).floor() as usize};
    let s = sketching_operator(DistributionType::Gaussian, d, a.nrows());
    let a_sk = &s*a;
    let b_sk = &s*b;
    let (q, r) = a_sk.qr().unpack();
    println!("b shape :({} {}), q shape: ({} {})",b_sk.nrows(), b_sk.ncols(), q.nrows(), q.ncols());
    let z_0 = q.transpose()*&b_sk;
    let rinv = r.try_inverse().unwrap();
    let a_preconditioned = a*&rinv;
    // TODO: Replace with Iterative solver
    let z = cgls(&a_preconditioned, &b, epsilon, l, &z_0);
    rinv*z
}

fn lsrn( a: &DMatrix<f64>, b: &DMatrix<f64>, sampling_factor: f64, epsilon: f64, l: usize) -> DMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();
    
    let d = (sampling_factor * n as f64).ceil() as usize;
    let s = sketching_operator(DistributionType::Gaussian, d, m);
    let a_sk = &s * a;

    let svd_obj = a_sk.svd(false, true);
    let sigma = DMatrix::from_diagonal(&svd_obj.singular_values);
    let v = svd_obj.v_t.unwrap().transpose();
    
    let mut sigma_inv = DMatrix::zeros(n, n);
    for i in 0..n {
        if sigma[i] > 0.0 {
            sigma_inv[(i, i)] = 1.0 / sigma[i];
        }
    }
    let n = v*&sigma_inv;
    let a_precond = a*&n;
    let mut y_hat = DMatrix::zeros(sigma_inv.ncols(), 1);
    y_hat = cgls(&a_precond, b, epsilon, l,  &y_hat);
    n* y_hat
}


pub fn lsrn_least_squares_overdetermined(a:&DMatrix<f64>, b:&DMatrix<f64>, epsilon:f64, l:usize, sampling_factor:f64) -> DMatrix<f64>
{
    let d = if sampling_factor*(a.ncols() as f64) > a.nrows() as f64 {a.nrows()} else {(sampling_factor*(a.ncols() as f64)).floor() as usize};
    let s = sketching_operator(DistributionType::Gaussian, d, a.nrows());
    let a_sk = &s*a;
    let b_sk = &s*b;
    let (q, r) = a_sk.qr().unpack();
    println!("b shape :({} {}), q shape: ({} {})",b_sk.nrows(), b_sk.ncols(), q.nrows(), q.ncols());
    let z_0 = q.transpose()*&b_sk;
    let rinv = r.try_inverse().unwrap();
    let a_preconditioned = a*&rinv;
    let z = cgls(&a_preconditioned, &b, epsilon, l, &z_0);
    rinv*z
}

// pub fn sketch_saddle_point_precondition(a:DMatrix<f64>, b:DMatrix<f64>, c:DMatrix<f64>, mu:f64, epsilon:f64, L:i32) -> DMatrix<f64>
// {

// }

#[cfg(test)]
mod tests
{
    use num_traits::Pow;
    use rand::Rng;
    use rand_distr::{Uniform, Normal, Distribution};
    use nalgebra::DMatrix;
    use crate::{sketch::{sketching_operator, DistributionType}, sketch_and_precondition::{blendenpik_least_squares_overdetermined, lsrn_least_squares_overdetermined}, solvers::solve_upper_triangular_system};
    use std::time::Instant;
    #[test]
    fn test_least_squares_qr()
    {
        // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
        let mut rng = rand::thread_rng();
        let n = rand::thread_rng().gen_range(10..30);
        let m = rand::thread_rng().gen_range(n..5000);
        let epsilon = 0.01;
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng));
        let mut data = sketching_operator(DistributionType::Gaussian, m, n);
        let y = &data*&hypothesis;
        for i in 0..m {
            let noise_vector = DMatrix::from_fn(n, 1, |_, _| normal.sample(&mut rng));
            for j in 0..n {
                data[(i, j)] += noise_vector[(j, 0)];
            }
        }
        let start1 = Instant::now();
        // compute using sketched qr
        let x = lsrn_least_squares_overdetermined(&data, &y, 0.0001, 10000, 4.0);
        let duration1 = start1.elapsed();
        // compute using plain qr
        let start2 = Instant::now();
        let (q, r) = data.qr().unpack();
        let b_transformed = q.transpose()*y;
        let actual_solution = solve_upper_triangular_system(&r, &b_transformed);
        let duration2 = start2.elapsed();
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        let mut norm3 = 0.0;
        let mut norm4 = 0.0;
        let mut norm5 = 0.0;
        let mut norm6 = 0.0;
        for i in 0..x.ncols(){
            norm1+= (hypothesis[(i, 0)] - actual_solution[(i, 0)]).pow(2.0);
            norm2+= (actual_solution[(i, 0)] - x[(i, 0)]).pow(2.0);
            norm3+= (x[(i, 0)] - hypothesis[(i, 0)]).pow(2.0);
            norm4+= (hypothesis[(i, 0)]).pow(2.0);
            norm5+= (actual_solution[(i, 0)]).pow(2.0);
            norm6+= (x[(i, 0)]).pow(2.0);
        }
        println!("{}, {}, {}, {}, {}, {}", norm1, norm2, norm3, norm4, norm5, norm6);
        println!("Time for sketched algorithm vs time for qr factorisation: {:.2?} {:.2?} ", duration1, duration2);
        // assert!((norm1 < 0.2));
        // assert!((norm2 < 0.2));
        // assert!((norm3 < 0.2));
        // println!("Hypothesis: \n{}", hypothesis);
        // println!("Unsketched Solution: \n{}", actual_solution);

    }
    
}