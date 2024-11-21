use nalgebra::{DMatrix, DVector};




/**
 * Input: Matrix A, a vector b, a tolerance value, a maximum number of iterations, and an optional initial guess x.
 * Output: The solution to the linear system Ax = b, or an error message if the method fails to converge.
 */
pub fn cgls(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    tolerance: f64,
    num_iterations: usize,
    x: Option<DMatrix<f64>>,
) -> DMatrix<f64> {
    // let m = a.nrows();
    let n = a.ncols();
    
    let mut x = x.unwrap_or_else(|| DMatrix::from_element(1, n, 0.0 as f64)); // Initial guess is zero
    let mut r = b - a * &x;              // Initial residual r = b - A * x
    let s = a.transpose() * &r;      // s = A^T * r
    let mut p = s.clone();               // Initial search direction
    let mut norm_s = s.dot(&s);          // Residual norm squared
    let mut converged = false;
    for i in 0..num_iterations {
        let ap = a * &p;                     // A * p
        let alpha = norm_s / ap.dot(&ap);    // Step size alpha
        x += alpha * &p;                     // Update solution x
        r -= alpha * ap;                     // Update residual r
        let s_new = a.transpose() * &r;      // s_new = A^T * r
        let norm_s_new = s_new.dot(&s_new);  // New residual norm squared

        // Convergence check based on tolerance
        if norm_s_new.sqrt() < tolerance {
            println!("CGLS converged after {} iterations", i + 1);
            converged = true;
            break;
        }

        let beta = norm_s_new / norm_s;      // Compute beta for next direction
        norm_s = norm_s_new;
        p = &s_new + beta * p;               // Update search direction
    }
    if !converged
    {
        println!("CGLS failed to converged after {} iterations", num_iterations);
    }
    x
}

pub fn conjugate_grad(a: &DMatrix<f64>, b: &DVector<f64>, x: Option<DVector<f64>>) -> DVector<f64> {
    let n = b.len();
    let mut x = x.unwrap_or_else(|| DVector::from_element(n, 1.0));
    let mut r = a * &x - b;
    let mut p = -&r;
    let mut r_k_norm = r.dot(&r);

    for i in 0..(2 * n) {
        let ap = a * &p;
        let alpha = r_k_norm / p.dot(&ap);
        x += alpha * &p;
        r += alpha * ap;
        let r_kplus1_norm = r.dot(&r);
        
        if r_kplus1_norm < 1e-10 {
            println!("Converged after {} iterations", i);
            break;
        }
        
        let beta = r_kplus1_norm / r_k_norm;
        r_k_norm = r_kplus1_norm;
        p = beta * p - &r;
    }

    x
}

pub fn verify_solution(a: &DMatrix<f64>, b: &DVector<f64>, x: &DVector<f64>) -> f64 {
    (a * x - b).norm()
}






mod test_conjugate_gradient {

    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_conjugate_gradient() {
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0]);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let x = DVector::from_row_slice(&[1.0, 1.0, 1.0]);
        let x_cg = conjugate_grad(&a, &b, Some(x));
        let error = verify_solution(&a, &b, &x_cg);
        assert!(error < 1e-10);
    }
}