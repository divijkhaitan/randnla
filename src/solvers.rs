#![allow(dead_code)]
#![allow(warnings)]
#![allow(unused_imports)]

use nalgebra::{DMatrix, Scalar, RealField, DVector, Dynamic, VecStorage, dmatrix, dvector};
use num_traits::{Zero, One};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};
use std::f64;



pub fn solve_upper_triangular_system<T>(u: &DMatrix<T>, y: &DMatrix<T>) -> DMatrix<T>
where
    T: Scalar + RealField + Zero + One + AddAssign + SubAssign + MulAssign + DivAssign + Copy,
{
    let n = u.nrows();
    let mut x = DMatrix::zeros(n, 1);
    
    for i in (0..n).rev() {
        if u[(i, i)] != T::zero()
        {
            let mut sum = T::zero();
            for j in (i+1)..n {
                sum += u[(i, j)] * x[(j, 0)];
            }
            
            x[(i, 0)] = (y[(i, 0)] - sum) / u[(i, i)];
        }
    }
    x
}

pub fn solve_diagonal_system<T>(u: &DMatrix<T>, y: &DMatrix<T>) -> DMatrix<T>
where
    T: Scalar + RealField + Zero + One + AddAssign + SubAssign + MulAssign + DivAssign + Copy,
{
    let n = u.nrows();
    let mut x = DMatrix::zeros(n, 1);
    
    for i in (0..n).rev() {
        if u[(i, i)] != T::zero()
        {
            x[(i, 0)] = y[(i, 0)] / u[(i, i)];
        }
    }
    x
}


// Stable implementation of Givens rotation

/* 
needed cause signum gives +1 for 0 which we don't want
*/
fn sign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } 
    else if x < 0.0 { -1.0 } 
    else { 0.0 }
}

fn sym_ortho(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        return (sign(a), 0.0, a.abs());
    } else if a == 0.0 {
        return (0.0, sign(b), b.abs());
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = sign(b) / (1.0 + tau * tau).sqrt();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c = sign(a) / (1.0 + tau * tau).sqrt();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

pub fn lsqr(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    damp: f64,
    atol: f64,
    btol: f64,
    conlim: f64,
    iter_lim: Option<usize>,
    calc_var: bool,
    x0: Option<&DVector<f64>>,
) -> (
    DVector<f64>,    // solution vector x
    usize,           // istop (reason for termination)
    usize,           // iterations performed
    f64,             // norm of residuals
    f64,             // norm including damping
    f64,             // estimated Frobenius norm of Abar
    f64,             // estimated condition number
    Vec<f64>,        // history of residual norms
    f64,             // norm of the solution
    DVector<f64>,    // variance estimate
) {
    let (m, n) = (a.nrows(), a.ncols());
    let eps = f64::EPSILON;
    let iter_lim = iter_lim.unwrap_or(2 * n);

    let mut var = DVector::zeros(n);
    let mut x = x0.cloned().unwrap_or_else(|| DVector::zeros(n));
    let mut u = b.clone();
    let bnorm = b.norm();

    // Adjust initial u if x0 is provided
    let mut beta = if x0.is_some() {
        u -= &(a * &x);
        u.norm()
    } else {
        bnorm
    };

    // Initialize bidiagonalization vectors
    u.scale_mut(if beta > 0.0 { 1.0 / beta } else { 0.0 });
    let mut v = a.transpose() * &u;
    let mut alfa = v.norm();
    
    v.scale_mut(if alfa > 0.0 { 1.0 / alfa } else { 0.0 });
    let mut w = v.clone();

    let mut rhobar = alfa;
    let mut phibar = beta;
    let mut rnorm = beta;
    let mut r1norm = rnorm;
    let mut r2norm = rnorm;

    let mut anorm: f64 = 0.0;
    let mut acond = 0.0;
    let dampsq = damp * damp;
    let mut ddnorm = 0.0;
    let mut res2 = 0.0;
    let mut xnorm = 0.0;
    let mut xxnorm = 0.0;
    let mut z = 0.0;
    let mut cs2 = -1.0;
    let mut sn2 = 0.0;

    let mut arnorm = alfa * beta;
    if arnorm == 0.0 {
        return (x, 0, 0, beta, beta, 0.0, 0.0, vec![0.0], 0.0, var);
    }

    let mut ctol = if conlim > 0.0 { 1.0 / conlim } else { 0.0 };
    let mut arnorms = vec![-1.0; iter_lim];
    let mut itn = 0;
    let mut istop = 0;

    while itn < iter_lim {
        arnorms[itn] = arnorm;
        itn += 1;

        // Bidiagonalization step
        u = a * &v - alfa * &u;
        beta = u.norm();

        if beta > 0.0 {
            u.scale_mut(1.0 / beta);
            anorm = (anorm.powi(2) + alfa.powi(2) + beta.powi(2) + dampsq).sqrt();
            
            v = a.transpose() * &u - beta * &v;
            alfa = v.norm();
            v.scale_mut(if alfa > 0.0 { 1.0 / alfa } else { 0.0 });
        }

        // Plane rotations
        let rhobar1 = (rhobar.powi(2) + dampsq).sqrt();
        let cs1 = rhobar / rhobar1;
        let sn1 = damp / rhobar1;
        let psi = sn1 * phibar;
        phibar *= cs1;

        let (cs, sn, rho) = sym_ortho(rhobar1, beta);

        let theta = sn * alfa;
        rhobar = -cs * alfa;
        let phi = cs * phibar;
        phibar *= sn;
        let tau = sn * phi;

        // Update x and w
        let t1 = phi / rho;
        let t2 = -theta / rho;
        let dk = w.scale(1.0 / rho);

        x += t1 * &w;
        w = &v + t2 * &w;
        ddnorm += dk.norm_squared();

        if calc_var {
            var += dk.component_mul(&dk);
        }

        // Estimate solution norm
        let delta = sn2 * rho;
        let gambar = -cs2 * rho;
        let rhs = phi - delta * z;
        let zbar = rhs / gambar;
        xnorm = (xxnorm + zbar.powi(2)).sqrt();
        let gamma = (gambar.powi(2) + theta.powi(2)).sqrt();
        cs2 = gambar / gamma;
        sn2 = theta / gamma;
        z = rhs / gamma;
        xxnorm += z.powi(2);

        // Convergence tests
        acond = anorm * ddnorm.sqrt();
        let res1 = phibar.powi(2);
        res2 += psi.powi(2);
        rnorm = (res1 + res2).sqrt();
        arnorm = alfa * tau.abs();

        let r1sq = rnorm.powi(2) - dampsq * xxnorm;
        r1norm = r1sq.abs().sqrt();
        r2norm = rnorm;

        let test1 = rnorm / bnorm;
        let test2 = arnorm / (anorm * rnorm + eps);
        let test3 = 1.0 / (acond + eps);
        let t1 = test1 / (1.0 + anorm * xnorm / bnorm);
        let rtol = atol + btol * (anorm * xnorm / bnorm);

        // Termination conditions
        if itn >= iter_lim { istop = 7; }
        if 1.0 + test3 <= 1.0 { istop = 6; }
        if 1.0 + test2 <= 1.0 { istop = 5; }
        if 1.0 + t1 <= 1.0 { istop = 4; }
        if test3 <= ctol { istop = 3; }
        if test2 <= atol { istop = 2; }
        if test1 <= rtol { istop = 1; }

        if istop != 0 { break; }
    }

    arnorms.retain(|&x| x > -1.0);
    
    (x, istop, itn, r1norm, r2norm, anorm, acond, arnorms, xnorm, var)
}




#[cfg(test)]
mod test_solvers {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simple_system() {
        // Create a simple 3x2 system: [[1, 0], [1, 1], [0, 1]]
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        
        // Test case 1: Zero solution
        let b = dvector![0.0, 0.0, 0.0];
        let (x, istop, _, _, _, _, _, _, _, _) = lsqr(&a, &b, 0.0, 1e-8, 1e-8, 1e8, None, false, None);

        println!("==== Zero soln x: {}", x);
        
        // assert_eq!(istop, 1); // Should converge
        
        // Test case 2: Known solution [1, -1]
        let b = dvector![1.0, 0.0, -1.0];
        let (x, istop, _, _, _, _, _, _, _, _) = lsqr(&a, &b, 0.0, 1e-8, 1e-8, 1e8, None, false, None);
        println!("==== Known soln x: {}", x);
        // assert_eq!(istop, 1); // Should converge
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(x[1], -1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_overdetermined_system() {

        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        /// Generates a random matrix of size (rows, cols) with normally distributed entries
        fn generate_random_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
            let normal = Normal::new(0.0, 1.0).unwrap();
            let mut rng = thread_rng();
            let data: Vec<f64> = (0..rows * cols).map(|_| normal.sample(&mut rng)).collect();
            DMatrix::from_vec(rows, cols, data)
        }

        /// Generates a random vector of size (size) with normally distributed entries
        fn generate_random_vector(size: usize) -> DVector<f64> {
            let normal = Normal::new(0.0, 1.0).unwrap();
            let mut rng = thread_rng();
            let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
            DVector::from_vec(data)
        }

        // Create an overdetermined system
        let a = DMatrix::from_row_slice(4, 2, &[
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0
        ]);
        let b = dvector![2.0, 4.0, 6.0, 8.0];

        let rows = 200;
        let cols = 100;
    
        // // Generate random matrix A and vector b
        let a = generate_random_matrix(rows, cols);
        let b = generate_random_vector(rows);

    
        // Solve the least squares problem using nalgebra's SVD
        let tick = std::time::Instant::now();
        let svd = a.clone().svd(true, true);
        let x_svd = svd.solve(&b, 1e-6).unwrap();
        println!("==== Time taken by nalgebra SVD: {:?}", tick.elapsed());
        // println!("==== x_svd: {}", x_svd);
        let residual_svd = &a * &x_svd - &b;
        println!("NALGEBRA residual norm svd: {:?}", residual_svd.norm());

        let tick = std::time::Instant::now();
        let (x, istop, _, _, _, _, _, _, _, _) = lsqr(&a, &b, 0.0, 1e-8, 1e-8, 1e8, None, false, None);
        // println!("==== x from lsqr: {}", x);
        println!("==== Time taken by lsqr: {:?}", tick.elapsed());
        
        // assert_eq!(istop, 1); // Should converge
        // The least squares solution should minimize ||Ax - b||
        // println!("==== x: {}", x);
        let residual = &a * &x - &b;
        println!("==== Residual norm LSQR: {:?}", residual.norm());
        // assert!(residual.norm() < 1e-6);

        println!("Residual norm difference: {:?}", (residual.norm() - residual_svd.norm()).abs());
    }

    #[test]
    fn test_damped_system() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let b = dvector![1.0, 1.0, 1.0];
        
        // Test with different damping parameters
        let (x1, _, _, _, _, _, _, _, _, _) = lsqr(&a, &b, 0.0, 1e-8, 1e-8, 1e8, None, false, None);
        let (x2, _, _, _, _, _, _, _, _, _) = lsqr(&a, &b, 1.0, 1e-8, 1e-8, 1e8, None, false, None);
        println!("==== x1: {}", x1);
        println!("==== x2: {}", x2);
        // Solution with damping should have smaller norm
        assert!(x2.norm() < x1.norm());
    }
}