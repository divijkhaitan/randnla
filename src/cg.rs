use nalgebra::{DMatrix, DVector};
use crate::errors::RandNLAError;

/**
* Inputs:
`a` is an `m x n` matrix,
`b` is an `m x 1` vector,
`x` is an `n x 1` initial guess vector,
`tolerance` is the tolerance value,
`num_iterations` is the maximum number of iterations

* Output:
The solution vector `x` that approximates the solution to the system Ax = b

This function uses the CGLS method to iteratively solve the system of linear equations. The method converges when the residual norm is below a specified tolerance or after a maximum number of iterations.
 */

pub fn cgls(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    tolerance: f64,
    num_iterations: usize,
    x: Option<DMatrix<f64>>,
) ->  DMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();

    // Initial guess
    let mut x = x.unwrap_or_else(|| DMatrix::from_element(n, 1 , 0.0 as f64));
    
    // Initial residual r = b - A * x
    let mut r = b - a * &x;              
    let s = a.transpose() * &r;
    let mut p = s.clone();
    let mut norm_s = s.dot(&s);
    let mut converged = false;
    for i in 0..num_iterations {
        let ap = a * &p;
        let alpha = norm_s / ap.dot(&ap);
        x += alpha * &p;
        r -= alpha * ap;
        let s_new = a.transpose() * &r;
        let norm_s_new = s_new.dot(&s_new);

        // check based on tolerance
        if norm_s_new.sqrt() < tolerance {
            println!("CGLS converged after {} iterations", i + 1);
            converged = true;
            break;
        }

        let beta = norm_s_new / norm_s;
        norm_s = norm_s_new;
        p = &s_new + beta * p;
    }
    if !converged
    {
        println!("CGLS failed to converged after {} iterations", num_iterations);
    }
    x
}


/**
Conjugate Gradient method for solving a system of linear equations Ax = b

* Inputs:
`a` is an `n x n` symmetric positive-definite matrix,
`b` is an `n x 1` vector,
`x` is an `n x 1` initial guess vector

* Output:
The solution vector `x` that approximates the solution to the system Ax = b
 */


pub fn conjugate_grad(a: &DMatrix<f64>, b: &DVector<f64>, x: Option<DVector<f64>>) -> Result<DVector<f64>, RandNLAError> {

    // Check positive semi-definite
    let eig = a.clone().symmetric_eigen();
    let eigvals = &eig.eigenvalues;
    if eigvals.iter().any(|&x| x < 0.0) {
        return Err(RandNLAError::NotPositiveSemiDefinite(
            "Matrix is not positive semi-definite".to_string()
        ));
    }
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

    Ok(x)
}



pub fn verify_solution(a: &DMatrix<f64>, b: &DVector<f64>, x: &DVector<f64>) -> f64 {
    (a * x - b).norm()
}






mod test_conjugate_gradient {

    use super::*;
    use crate::test_assist::{self, generate_random_matrix};


    #[test]
    fn test_cgls_converges() {
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 2.0,
            1.0, 3.0, 0.0,
            2.0, 0.0, 1.0
        ]);
        let b = DMatrix::from_row_slice(3, 1, &[4.0, 2.0, 2.0]);
        let tolerance = 3.0;
        let num_iterations = 100;
        let x = None;

        let result = cgls(&a, &b, tolerance, num_iterations, x);
        let expected = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 0.0]); // Expected result should be calculated
        let error= result - expected;
        println!("result-exp: {:?}", error.norm());

        assert!(error.norm() < tolerance);
    }

    #[test]
    fn test_cgls_with_initial_guess() {
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 2.0,
            1.0, 3.0, 0.0,
            2.0, 0.0, 1.0
        ]);
        let b = DMatrix::from_row_slice(3, 1, &[4.0, 2.0, 2.0]);
        let tolerance = 3.0;
        let num_iterations = 100;
        let initial_guess = DMatrix::from_row_slice(3, 1, &[1.0, 1.0, 1.0]);

        let result = cgls(&a, &b, tolerance, num_iterations, Some(initial_guess));
        let expected = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 0.0]); // Expected result should be calculated
    
        assert!((result - expected).norm() < tolerance);
    }

    #[test]
    fn test_cgls_does_not_converge() {
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 2.0,
            1.0, 3.0, 0.0,
            2.0, 0.0, 1.0
        ]);
        let b = DMatrix::from_row_slice(3, 1, &[4.0, 2.0, 2.0]);
        let tolerance = 1e-20;
        let num_iterations = 1;
        let x = None;

        let result = cgls(&a, &b, tolerance, num_iterations, x);
        let expected = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 0.0]); // Expected result should be calculated

        assert!((result - expected).norm() > tolerance);
    }

    #[test]
    fn test_conjugate_gradient() {
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0]);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let x = DVector::from_row_slice(&[1.0, 1.0, 1.0]);
        let x_cg = conjugate_grad(&a, &b, Some(x));
        assert!(x_cg.is_ok());
        let error = verify_solution(&a, &b, &x_cg.unwrap());
        println!("error: {:?}", error);
        assert!(error < 1e-10);
    }

    
}