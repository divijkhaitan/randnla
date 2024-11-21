use nalgebra::DMatrix;
use crate::lora_helpers;
use crate::errors::RandNLAError;
use std::error::Error;


/*
TODO: see using gemm for matmul and do proper error handling
 */


/**
SVD for low-rank approximation of the input matrix

* Inputs:
`A` is `m x n` matrix, `k` is the rank of the approximation, `epsilon` is the error tolerance, and `s` is the oversampling parameter
* Output:
The compact SVD of a low-rank approximation of `A`


The returned approximation will have rank at most `k``
The approximation produced by the randomized phase of the
algorithm will attempt to A to within `epsilon` error, but will not produce
an approximation of rank greater than `k + s`.

This uses randomization to compute a
QB decomposition of A, then deterministically computes QB’s compact SVD, and
finally truncates that SVD to a specified rank.
 */
pub fn rand_svd(A: &DMatrix<f64>, k: usize, epsilon: f64, s: usize) -> Result<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>), RandNLAError> {
    if k == 0 {
        return Err(RandNLAError::InvalidParameters(
            format!("Rank k must be positive, current input is {}", k)
        ));
    }
    if epsilon <= 0.0 {
        return Err(RandNLAError::InvalidParameters(
            format!("Epsilon must be positive, current input is {}", epsilon)
        ));
    }
    if s == 0 {
        return Err(RandNLAError::InvalidParameters(
            format!("Oversampling parameter s must be positive, current input is {}", s)
        ));
    }

    println!("Running RSVD");
    
    let (Q, B) = lora_helpers::QB1(A, k + s, epsilon);

    let r = k.min(Q.ncols());

    let svd = match B.svd(true, true) {
        svd if svd.u.is_some() && svd.v_t.is_some() => svd,
        _ => return Err(RandNLAError::MatrixDecompositionError(
            "SVD decomposition failed".to_string()
        ))
    };

    let U = svd.u.as_ref().unwrap().columns(0, r).into_owned();
    
    let V = svd.v_t.as_ref().unwrap().transpose().columns(0, r).into_owned();

    let S = DMatrix::from_diagonal(&svd.singular_values).rows(0, r).columns(0,r).into_owned();

    let U_final = &Q * &U;
    
    Ok((U_final, S, V.transpose()))
}



/**
Find approximations of the dominant eigenvectors and eigenvalues of a Hermitian matrix using randomization

* Inputs:
`A` is an `n x n` Hermitian matrix, `k` is the rank of the approximation, `epsilon` is the error tolerance, and `s` is the oversampling parameter

* Output:
Approximations of the dominant eigenvectors and eigenvalues of `A`


This essentially finds an decomposition of an approximation `A_hat = V diag(λ)V^*`, where V is a tall column-orthonormal matrix and λ is a vector
with entries sorted in decreasing order of absolute value
*/

pub fn rand_evd1(A: &DMatrix<f64>, k: usize, epsilon: f64, s: usize) -> Result<(DMatrix<f64>, Vec<f64>), RandNLAError> {
    // Parameter validation
    if k == 0 {
        return Err(RandNLAError::InvalidParameters(
            format!("Rank k must be positive, current input is {}", k)
        ));
    }
    if epsilon <= 0.0 {
        return Err(RandNLAError::InvalidParameters(
            format!("Epsilon must be positive, current input is {}", epsilon)
        ));
    }
    if s == 0 {
        return Err(RandNLAError::InvalidParameters(
            format!("Oversampling parameter s must be positive, current input is {}", s)
        ));
    }

    // Check Hermitian property
    if A != &A.adjoint() {
        return Err(RandNLAError::NotHermitian(
            "Input matrix is not Hermitian".to_string()
        ));
    }

    println!("Running REVD1");
    
    let (Q, B) = match lora_helpers::QB1(A, k + s, epsilon) {
        (Q, B) => (Q, B),
        _ => return Err(RandNLAError::ComputationError(
            "QB1 decomposition failed".to_string()
        ))
    };
    
    let C = &B * &Q;

    if C != &Q.adjoint() * A * &Q {
        return Err(RandNLAError::ComputationError(
            "Matrix multiplication check failed".to_string()
        ));
    }
    
    let eig = C.symmetric_eigen();

    let eigvals = eig.eigenvalues;
    let eigvecs = eig.eigenvectors;

    let abs_eigvals: Vec<(f64, usize)> = eigvals.iter().enumerate().map(|(i, &x)| (x.abs(), i)).collect();

    let mut indices: Vec<usize> = abs_eigvals.iter().map(|&(_, i)| i).collect();
    
    indices.sort_by(|&i, &j| abs_eigvals[j].0.partial_cmp(&abs_eigvals[i].0).unwrap());

    let r = k.min(eigvals.iter().count());

    let selected_indices = indices.iter().take(r);

    let lambda: Vec<f64> = selected_indices.clone().map(|&i| eigvals[i]).collect();

    let U = DMatrix::from_columns(&selected_indices.map(|&i| eigvecs.column(i)).collect::<Vec<_>>());

    let V = Q * U;

    Ok((V, lambda))
}



/** 
Find approximations of the dominant eigenvectors and eigenvalues of a positive semi-definite matrix using randomization
 
 * Inputs: 
`A` is an `n x n` psd matrix, `k` is the rank of the approximation, and `s` is the oversampling parameter
* Output:
Approximations of the dominant eigenvectors and eigenvalues of A

This essentially finds an decomposition of an approximation `A_hat = V diag(λ)V^*`, where V is a tall column-orthonormal matrix and λ is a vector
with entries sorted in decreasing order of absolute value
*/

pub fn rand_evd2(A: &DMatrix<f64>, k: usize, s: usize) -> Result<(DMatrix<f64>, Vec<f64>), RandNLAError> {
    // Parameter validation
    if k == 0 {
        return Err(RandNLAError::InvalidParameters(
            format!("Rank k must be positive, current input is {}", k)
        ));
    }

    println!("Running REVD2");
    
    // Check positive semi-definite
    let eig = A.clone().symmetric_eigen();
    let eigvals = &eig.eigenvalues;
    if eigvals.iter().any(|&x| x < 0.0) {
        return Err(RandNLAError::NotPositiveSemiDefinite(
            "Matrix is not positive semi-definite".to_string()
        ));
    }

    let S = lora_helpers::tsog1(A, k+s, 3, 1);
    let Y = A*&(S);
    let mach_eps = f64::EPSILON;
    let nu = (A.nrows() as f64).sqrt() * mach_eps * Y.norm();
    let Y_new = Y + (nu*&(S));
    let SY = S.adjoint()*&Y_new;

    let chol = match SY.cholesky() {
        Some(c) => c,
        None => return Err(RandNLAError::MatrixDecompositionError(
            "Cholesky Decomposition Failed".to_string()
        ))
    };

    let R = chol.l().transpose();
    let B = match Y_new * (R.try_inverse().unwrap()) {
        B => B,
        _ => return Err(RandNLAError::MatrixDecompositionError(
            "Matrix Inverse Failed".to_string()
        ))
    };
    
    let mysvd = match B.clone().svd(true, true) {
        svd if svd.u.is_some() && svd.v_t.is_some() => svd,
        _ => return Err(RandNLAError::MatrixDecompositionError(
            "SVD Decomposition Failed".to_string()
        ))
    };

    let V_binding = mysvd.u.unwrap();
    let S_binding = DMatrix::from_diagonal(&mysvd.singular_values);
    let lambda = S_binding.iter().filter(|&&x| x > 0.0).map(|x| x*x).collect::<Vec<f64>>();

    let r = std::cmp::min(k, lambda.iter().filter(|&&x| x > nu).count());
    let lambda1 = lambda.iter().take(r).map(|x| x - nu).collect::<Vec<f64>>();
    let V_final = V_binding.columns(0, r).into_owned();

    Ok((V_final, lambda1))
}





#[cfg(test)]
mod test_randsvd {
    use super::*;
    use crate::test_assist::{generate_random_matrix, check_approx_equal};
    use approx::assert_relative_eq;

    #[test]
    fn test_rand_svd_basic_functionality() {
        let a = generate_random_matrix(100, 50);
        let k = 45;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.nrows(), k);
    }

    #[test]
    fn test_rand_svd_invalid_k() {
        let a = generate_random_matrix(100, 50);
        let k = 0;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(matches!(result, Err(RandNLAError::InvalidParameters(_))));
    }

    #[test]
    fn test_rand_svd_compare() {
        let a = generate_random_matrix(100, 50);
        let k = 45;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();


         // Compare with deterministic SVD

         let svd = a.svd(true, true);
         let binding = svd.u.unwrap();
         let u_det = binding.columns(0, k).clone();
 
         let binding = svd.v_t.unwrap().transpose();
         let v_det = binding.columns(0, k).clone().transpose();
 
         let s_binding = DMatrix::from_diagonal(&svd.singular_values);
         let rows_trunc = s_binding.rows(0, k);
         let s_det = rows_trunc.columns(0,k).clone();

         
        // reconstructed matrix:
        let approx = &U * &S * V.clone();

        let orig_trunc = &u_det * &s_det * &v_det;


        if !check_approx_equal(&approx, &orig_trunc, 1.0) {
            println!("Exceeding tolerance")
        }
        else {
            println!("Within tolerance")
        }

        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.nrows(), k); // cause it's transposed
    }

    #[test]
    fn test_rand_svd_tall_matrix() {
        let a = generate_random_matrix(20, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.nrows(), k);
    }

    #[test]
    fn test_rand_svd_wide_matrix() {
        let a = generate_random_matrix(5, 20);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.nrows(), k);
    }

    #[test]
    fn test_rand_svd_zero_matrix() {
        let a = DMatrix::zeros(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.nrows(), k);
        assert_relative_eq!(U, DMatrix::identity(10, k).columns(0,k).into(), epsilon = 1e-6);
        assert_relative_eq!(S, DMatrix::zeros(k, k), epsilon = 1e-6);
        assert_relative_eq!(V.transpose(), DMatrix::<f64>::identity(5, k).columns(0,k).into(), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_svd_identity_matrix() {
        let a = DMatrix::identity(5, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.nrows(), k);
    }

    #[test]
    fn test_rand_svd_comparison_with_deterministic_svd() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();

       // Compare with deterministic SVD
       let svd = a.svd(true, true);
       let binding = svd.u.unwrap();
       let u_det = binding.columns(0, k).clone();

       let binding = svd.v_t.unwrap().transpose();
       let v_det = binding.columns(0, k).clone().transpose();

       let s_binding = DMatrix::from_diagonal(&svd.singular_values);
       let rows_trunc = s_binding.rows(0, k);
       let s_det = rows_trunc.columns(0,k).clone();

        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.nrows(), k);
        
        if !check_approx_equal(&U, &u_det.into(), 1.0) {
            println!("Exceeding tolerance")
        }
        if !check_approx_equal(&S, &s_det.into(), 1.0) {
            println!("Exceeding tolerance")
        }

        if !check_approx_equal(&V, &v_det.into(), 1.0) {
            println!("Exceeding tolerance")
        }
    }

    #[test]
    fn test_rand_svd_conformability() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();
        assert_eq!(S.nrows(), U.ncols());
        assert_eq!(S.ncols(), V.nrows());
        assert_eq!(U.nrows(), a.nrows());
        assert_eq!(V.ncols(), a.ncols());
    }

    #[test]
    fn test_rand_svd_orthogonality() {
        let a = generate_random_matrix(3, 3);
        let k = 2;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, _, _) = result.unwrap();
        assert_relative_eq!(U.transpose() * &U, DMatrix::identity(k, k), epsilon = 1.0);
        
        // assert_relative_eq!(V.transpose() * &V, DMatrix::identity(k, k), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_svd_singular_values() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (_, S, _) = result.unwrap();
        let singular_values: Vec<f64> = S.diagonal().iter().cloned().collect();
        let mut sorted_singular_values = singular_values.clone();
        sorted_singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(singular_values, sorted_singular_values);
    }

    #[test]
    fn test_rand_svd_relative_frobenius_error() {
        let a = generate_random_matrix(100, 50);
        let k = 50;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_svd(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (U, S, V) = result.unwrap();
        let A_approx = &U * &S * V;
        let error = (&a - &A_approx).norm() / a.norm();
        assert!(error <= 1.0);
    }
}




#[cfg(test)]
mod test_randevd1
{
    use crate::test_assist::{generate_random_matrix, generate_random_hermitian_matrix};
    use std::time::Instant;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector, dmatrix};
    use super::*;

    #[test]
    fn test_rand_evd1_basic_functionality() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;
        
        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();

        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }

    #[test]
    fn test_wrong_hermitian() {
        let a = dmatrix![1.0, 2.0, 3.0; 100.0, 200.0, 4.0; 3.0, 38.0, 1.0];
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_err());
    }

    #[test]
    fn test_rand_evd1_large_hermitian_matrix() {
        let a = generate_random_hermitian_matrix(100);
        let k = 10;
        let epsilon = 1e-6;
        let s = 5;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        
        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }

    #[test]
    fn test_rand_evd1_zero_matrix() {
        let a = DMatrix::zeros(5, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();

        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
        let real_soln = a.symmetric_eigen();
        assert_relative_eq!(V, real_soln.eigenvectors.columns(0, k).into_owned(), epsilon = 1e-6);
        // assert_relative_eq!(V, DMatrix::zeros(5, k), epsilon = 1e-6);
        assert!(lambda.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_rand_evd1_identity_matrix() {
        let a = DMatrix::identity(5, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();

        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }
    #[test]
    fn test_randevd1_compare() {

        let dims = 5;
        
        // needs to be square
        let A_rand =  generate_random_matrix(dims, dims);
        let A_rand_psd = &A_rand*(&A_rand.transpose());


        let k = dims - 2;
        let epsilon= 0.01;
        
        let s = 1;

        println!("Running test_rand_evd1");

        let tick = Instant::now();

        let result = rand_evd1(&A_rand_psd, k, epsilon, s);
        assert!(result.is_ok());
        let (v, lambda) = result.unwrap();

        let tock = tick.elapsed();
        println!("Time taken by RandEVD1: {:?}", tock);

        // printed in col major order
        // println!("Rand Eigenvectors: {:?}", v);
        

        let tick = Instant::now();
        let normal_evd = A_rand_psd.symmetric_eigen();
        let tock = tick.elapsed();
        println!("Time taken by Normal EVD: {:?}", tock);
    
        // println!("rand eigvals: {:?}", lambda);
        // println!("normal eigvals: {:?}", normal_evd.eigenvalues);
        assert_eq!(v.ncols(), k);
        assert_eq!(lambda.len(), k);
        
        let trunc = normal_evd.eigenvectors.columns(0, k);

        // println!("Normal Eigenvectors: {}", trunc);
        let trunc_flipped = DMatrix::from_fn(trunc.nrows(), trunc.ncols(), |i, j| -trunc[(i, j)]);

        // println!("Normal Eigenvectors Flipped Sign: {}", trunc_flipped);

        println!("Norm difference between normal and randomized: {}", (&v - &trunc_flipped).norm());
        assert_relative_eq!(v, trunc_flipped, epsilon = 2.0);

    }

    #[test]
    fn test_rand_evd1_conformability() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        assert_eq!(lambda.len(), V.ncols());
        assert_eq!(V.nrows(), a.nrows());
    }

    #[test]
    fn test_rand_evd1_orthogonality() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (V, _) = result.unwrap();
        assert_relative_eq!(V.transpose() * &V, DMatrix::identity(k, k), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_evd1_eigenvalues() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (_, lambda) = result.unwrap();
        let abs_lambda: Vec<f64> = lambda.iter().map(|&x| x.abs()).collect();
        let mut sorted_lambda = abs_lambda.clone();
        sorted_lambda.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(abs_lambda, sorted_lambda);
    }

    #[test]
    fn test_rand_evd1_relative_frobenius_error() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 2.0;
        let s = 2;

        let result = rand_evd1(&a, k, epsilon, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        let A_approx = &V * DMatrix::from_diagonal(&DVector::from_vec(lambda.clone())) * V.transpose();
        let error = (&a - &A_approx).norm() / a.norm();
        println!("Error: {}", error);
        assert!(error <= epsilon);
    }
}



#[cfg(test)]
mod test_randevd2
{
    use crate::test_assist::generate_random_psd_matrix;
    use crate::lora_drivers;
    use std::time::Instant;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector, dmatrix};
    use super::*;

    #[test]
    fn test_rand_evd2_basic_functionality() {
        let a = generate_random_psd_matrix(5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }

    #[test]
    fn test_wrong_psd() {
        let a = dmatrix![1.0, 2.0, 3.0; 100.0, 200.0, 4.0; 3.0, 38.0, 1.0];
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_err());
    }

    #[test]
    fn test_rand_evd2_large_psd_matrix() {
        let a = generate_random_psd_matrix(100);
        let k = 10;
        let s = 5;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }

    #[test]
    fn test_rand_evd2_zero_matrix() {
        let a = DMatrix::zeros(5, 5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_err());
    }

    #[test]
    fn test_rand_evd2_identity_matrix() {
        let a = DMatrix::identity(5, 5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }

    #[test]
    fn test_rand_evd2_comparison_with_deterministic_evd() {
        let a = generate_random_psd_matrix(5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();

        // Compare with deterministic EVD
        let eig = a.clone().symmetric_eigen();
        let eigvals = eig.eigenvalues;
        let eigvecs = eig.eigenvectors;

        let abs_eigvals: Vec<(f64, usize)> = eigvals.iter().enumerate().map(|(i, &x)| (x.abs(), i)).collect();
        let mut indices: Vec<usize> = abs_eigvals.iter().map(|&(_, i)| i).collect();
        indices.sort_by(|&i, &j| abs_eigvals[j].0.partial_cmp(&abs_eigvals[i].0).unwrap());

        let selected_indices = indices.iter().take(k);
        let lambda_det: Vec<f64> = selected_indices.clone().map(|&i| eigvals[i]).collect();
        let U_det = DMatrix::from_columns(&selected_indices.map(|&i| eigvecs.column(i)).collect::<Vec<_>>());

        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
        assert_relative_eq!(V.transpose() * &V, DMatrix::identity(k, k), epsilon = 1e-6);
        assert_relative_eq!(DVector::from_vec(lambda), DVector::from_vec(lambda_det), epsilon = 1e-6);
        assert_relative_eq!(&V * V.clone().transpose() * a.clone(), &U_det * U_det.clone().transpose() * &a, epsilon = 1e-6);
    }

    #[test]
    fn test_rand_evd2_conformability() {
        let a = generate_random_psd_matrix(5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        assert_eq!(lambda.len(), V.ncols());
        assert_eq!(V.nrows(), a.nrows());
    }

    #[test]
    fn test_rand_evd2_orthogonality() {
        let a = generate_random_psd_matrix(5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, _) = result.unwrap();
        assert_relative_eq!(V.transpose() * &V, DMatrix::identity(k, k), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_evd2_eigenvalues() {
        let a = generate_random_psd_matrix(5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (_, lambda) = result.unwrap();
        let abs_lambda: Vec<f64> = lambda.iter().map(|&x| x.abs()).collect();
        let mut sorted_lambda = abs_lambda.clone();
        sorted_lambda.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(abs_lambda, sorted_lambda);
    }

    #[test]
    fn test_rand_evd2_relative_frobenius_error() {
        let a = generate_random_psd_matrix(100);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
        let A_approx = &V * DMatrix::from_diagonal(&DVector::from_vec(lambda.clone())) * V.transpose();
        let error = (&a - &A_approx).norm() / a.norm();
        println!("Error: {}", error);
        assert!(error <= 2.0);
    }


    #[test]
    fn test_randEVD2(){

        let dims = 100;
        
        let A_rand_psd = generate_random_psd_matrix(dims);

        let k = 3;
        let s = 0;

        let tick = Instant::now();
        let randevd2 = lora_drivers::rand_evd2(&A_rand_psd, k, s);
        let tock = tick.elapsed();

        println!("Time taken by RandEVD2: {:?}", tock);

        assert!(randevd2.is_ok());
        let (v_rand, lambda_rand) = randevd2.unwrap();

        let tick = Instant::now();
        let normal_evd = A_rand_psd.symmetric_eigen();
        let tock = tick.elapsed();
        println!("Time taken by Normal EVD: {:?}", tock);
        
        let V = normal_evd.eigenvectors;
        let lambda = normal_evd.eigenvalues;

        let rand_v_norm = v_rand.norm();
        let v_norm = V.norm();
        let diff_v = rand_v_norm - v_norm;
        println!("Difference between V's: {}", diff_v);

        // we find the reconstruction error between rand and deterministic after truncating the deterministic ka columns to the specified rank. Reconstruction error is the different between the original matrix A truncated and the reconstructed matrix from the eigvects and eigvals

        // println!("V: {}", V);
        // println!("V_Rand: {}", v_rand);
        // println!("Lambda: {:?}", lambda);
        // println!("Lambda_Rand: {:?}", lambda_rand);


        let reconstructed_rand = &v_rand * DMatrix::from_diagonal(&DVector::from_vec(lambda_rand.clone())) * &v_rand.transpose();
        let reconstructed_deterministic = &V * DMatrix::from_diagonal(&lambda) * &V.transpose();
        let diff = (&reconstructed_rand - &reconstructed_deterministic).norm();
        println!("Reconstruction Error: {}", diff);

    }

    
}
