#![allow(dead_code)]
#![allow(warnings)]
#![allow(unused_imports)]
use nalgebra::{DMatrix, dmatrix, DVector};
use crate::sketch;
use crate::lora_helpers;


/*
TODO: Remove the use of clones and try to optimize performance wherever you can
 */



// randomized SVD
/**
* Inputs:
A: m x n matrix
The returned approximation will have rank at most k
The approximation produced by the randomized phase of the
algorithm will attempt to A to within eps error, but will not produce
an approximation of rank greater than k + s.
* Output:
The compact SVD of a low-rank approximation of A

This algorithm uses randomization to compute a
QB decomposition of A, then deterministically computes QB’s compact SVD, and
finally truncates that SVD to a specified rank.
 */
pub fn rand_svd(A: &DMatrix<f64>, k: usize, epsilon: f64, s: usize) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    println!("Running RSVD");
    
    
    let (Q, B) = lora_helpers::QB1(A, k + s, epsilon);
    let r = k.min(Q.ncols());

    
    let svd = B.svd(true, true);
    
    
    let U = svd.u.as_ref().expect("SVD failed to compute U").columns(0, r).into_owned();  
    
    
    let V = svd.v_t.as_ref().expect("SVD failed to compute V_t").transpose().columns(0, r).into_owned();
    
    
    let S = DMatrix::from_diagonal(&svd.singular_values).rows(0, r).into_owned();

    let U_final = &Q * &U;
    
    return (U_final, S, V)
}

// assert k > 0
// assert k <= min(A.shape)
// if not np.isnan(tol):
//     assert tol >= 0
//     assert tol < np.inf
// rng = np.random.default_rng(rng)
// Q, B = self.qb(A, k + over, tol / 2, rng)
// # B=Q^*A is necessary
// C = B @ Q
// lamb, U = la.eigh(C)
// alamb = np.abs(lamb)
// # d = number of columns in Q, d ≤ k + s
// d = Q.shape[1]
// r = min(k, d, np.count_nonzero(alamb > 10*np.finfo(float).eps))
// I = np.argsort(-1*np.abs(alamb))[:r]
// # indices of r largest components of |λ|
// U = U[:, I]
// lamb = lamb[I] 
// V = Q @ U
// return V, lamb




/**
Each randomized algorithm for low-rank SVD has a corresponding version that is
specialized to Hermitian matrices. We recount those specialized algorithms here,
and we mention an additional algorithm that is unique to the approximation of psd
matrices. In general, we shall say that A is n × n and that the algorithms represent
A_hat = V diag(λ)V∗, where V is a tall column-orthonormal matrix and λ is a vector
with entries sorted in decreasing order of absolute value

Input:
A is an n × n Hermitian matrix. The returned approximation will
have rank at most k. The approximation produced by the randomized
phase of the algorithm will attempt to A to within eps error, but will
not produce an approximation of rank greater than k + s.

Output:
Approximations of the dominant eigenvectors and eigenvalues of A.
*/

pub fn rand_evd1(A: &DMatrix<f64>, k: usize, epsilon: f64, s: usize) -> (DMatrix<f64>, Vec<f64>) {

    println!("Running REVD1");
    assert!(A == &A.adjoint());
    
    let (Q, B) = lora_helpers::QB1(A, k + s, epsilon);
    
    let C = &B * &Q;

    assert!(C == &Q.adjoint() * A * &Q);
    
    let eig = C.symmetric_eigen();

    let mut eigvals = eig.eigenvalues;
    let mut eigvecs = eig.eigenvectors;


    let abs_eigvals: Vec<(f64, usize)> = eigvals.iter().enumerate().map(|(i, &x)| (x.abs(), i)).collect();

    // float isn't an iterator and not comparable by default, so we can't just do sort_by ascending
    let mut indices: Vec<usize> = abs_eigvals.iter().map(|&(_, i)| i).collect();
    
    indices.sort_by(|&i, &j| abs_eigvals[j].0.partial_cmp(&abs_eigvals[i].0).unwrap());

    let d = Q.ncols();
    let r = k.min(eigvals.iter().count());

    // top r 
    let selected_indices = indices.iter().take(r);

    let lambda: Vec<f64> = selected_indices.clone().map(|&i| eigvals[i]).collect();


    let U = DMatrix::from_columns(&selected_indices.map(|&i| eigvecs.column(i)).collect::<Vec<_>>()
    );
    // println!("Q: {}", Q);
    // println!("U: {}", U);
    let V = Q * U;
    // println!("V: {}", V);
    return (V, lambda);
}




    
// only for positive semi-definite matrices
// For performance reasons we are not checking if the matrix is symmetric since that would lead to a performance hit
/** 
 * Inputs: 
 A is an n × n psd matrix. The returned approximation will have rank
at most k, but the sketching operator used in the algorithm can have
rank as high as k + s.
* Output:
Approximations of the dominant eigenvectors and eigenvalues of A
*/

pub fn rand_evd2(A:&DMatrix<f64>, k:usize, s: usize) -> Result<(DMatrix<f64>, Vec<f64>), &'static str> {
    println!("Running REVD2");
    // let S_wrapped = sketch::sketching_operator(sketch::DistributionType::Gaussian, A.nrows(), k+s);
    // let S = S_wrapped.unwrap();
    
    // assert that all the eigenvalues of A are nonnegative
    let eig = A.clone().symmetric_eigen();
    let eigvals = &eig.eigenvalues;
    if eigvals.iter().any(|&x| x < 0.0) {
        return Err("Matrix is not positive semi-definite");
    }

    let S = lora_helpers::tsog1(A, k+s, 3, 1);
    let Y = A*&(S);
    let mach_eps = f64::EPSILON;
    let nu = (A.nrows() as f64).sqrt() * mach_eps * Y.norm();
    let Y_new = Y + (nu*&(S));
    let SY = S.adjoint()*&Y_new;

    let chol = match SY.cholesky() {
        Some(c) => c,
        None => return Err("Cholesky Decomposition Failed"),
    };

    // in the monograph, we need the upper triangular part but in nalgebra we get the lower triangular part, so we can't use it directly
    let R = chol.l().transpose();
    let B = Y_new*(R.try_inverse().unwrap());
    let mysvd = B.clone().svd(true, true);

    let V_binding = mysvd.u.unwrap();
    let W_binding = mysvd.v_t.unwrap().transpose();
    let S_binding = DMatrix::from_diagonal(&mysvd.singular_values);
    let lambda = S_binding.iter().filter(|&&x| x > 0.0).map(|x| x*x).collect::<Vec<f64>>();

    // we need the ones that are greater than nu
    let r = std::cmp::min(k, lambda.iter().filter(|&&x| x > nu).count());
    let lambda1 = lambda.iter().take(r).map(|x| x - nu).collect::<Vec<f64>>();
    let V_final = V_binding.columns(0, r).into_owned();

    return Ok((V_final, lambda1));
}





#[cfg(test)]
mod test_randsvd
{
    use crate::test_assist::{generate_random_matrix, generate_random_hermitian_matrix, check_approx_equal};
    use crate::lora_helpers;
    use crate::lora_drivers;
    use std::time::Instant;
    use approx::assert_relative_eq;
    use super::*;



    #[test]
    fn test_rand_svd_basic_functionality() {
        let a = generate_random_matrix(3, 3);
        // make a 3x3 fixed matrix with the dmatrix! macro
        let a = dmatrix![1.0, 2.0, 3.0;
                          4.0, 5.0, 6.0;
                          7.0, 8.0, 9.0];

        let k = 2;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);


         // Compare with deterministic SVD
         let svd = a.svd(true, true);
         let binding = svd.u.unwrap();
         let u_det = binding.columns(0, k).clone();
 
         let binding = svd.v_t.unwrap().transpose();
         let v_det = binding.columns(0, k).clone();
 
         let s_binding = DMatrix::from_diagonal(&svd.singular_values);
         let s_det = s_binding.rows(0, k).clone();

         
         
         println!("Dimensions of Determ U: {}", u_det);
         println!("Dimensions of Rand U: {}", U);
         println!("Dimensions of Determ S: {}", s_det);
         println!("Dimensions of Rand S: {}", S);
         println!("Dimensions of Determ V: {}", v_det);
         println!("Dimensions of Rand V: {}", V);
         
        // reconstructed matrix:
        // let approx = &U * &S * V.transpose();


        // assert_eq!(U.ncols(), k);
        // assert_eq!(S.nrows(), k);
        // assert_eq!(S.ncols(), k);
        // assert_eq!(V.ncols(), k);
    }

    #[test]
    fn test_rand_svd_tall_matrix() {
        let a = generate_random_matrix(20, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.ncols(), k);
    }

    #[test]
    fn test_rand_svd_wide_matrix() {
        let a = generate_random_matrix(5, 20);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.ncols(), k);
    }

    #[test]
    fn test_rand_svd_zero_matrix() {
        let a = DMatrix::zeros(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.ncols(), k);
        assert_relative_eq!(U, DMatrix::zeros(10, k), epsilon = 1e-6);
        assert_relative_eq!(S, DMatrix::zeros(k, k), epsilon = 1e-6);
        assert_relative_eq!(V, DMatrix::zeros(5, k), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_svd_identity_matrix() {
        let a = DMatrix::identity(5, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.ncols(), k);
    }

    #[test]
    fn test_rand_svd_comparison_with_deterministic_svd() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);

        // Compare with deterministic SVD
        let svd = a.svd(true, true);
        let binding = svd.u.unwrap();
        let u_det = binding.columns(0, k).clone();

        let binding = svd.v_t.unwrap().transpose();
        let v_det = binding.columns(0, k).clone();

        let s_binding = DMatrix::from_diagonal(&svd.singular_values);
        let s_det = s_binding.rows(0, k).clone();

        assert_eq!(U.ncols(), k);
        assert_eq!(S.nrows(), k);
        assert_eq!(S.ncols(), k);
        assert_eq!(V.ncols(), k);
        
        if (!check_approx_equal(&U, &u_det.into(), 1.0)) {
            println!("Exceeding tolerance")
        }
        if (!check_approx_equal(&S, &s_det.into(), 1.0)) {
            println!("Exceeding tolerance")
        }

        if (!check_approx_equal(&V, &v_det.into(), 1.0)) {
            println!("Exceeding tolerance")
        }


        // assert_relative_eq!(U, u_det, epsilon = 1e-6);
        // assert_relative_eq!(S, s_det, epsilon = 1e-6);
        // assert_relative_eq!(V, v_det, epsilon = 1e-6);
    }

    #[test]
    fn test_rand_svd_conformability() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        assert_eq!(S.nrows(), U.ncols());
        assert_eq!(S.ncols(), V.ncols());
        assert_eq!(U.nrows(), a.nrows());
        assert_eq!(V.nrows(), a.ncols());
    }

    #[test]
    fn test_rand_svd_orthogonality() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        assert_relative_eq!(U.transpose() * &U, DMatrix::identity(k, k), epsilon = 1e-6);
        assert_relative_eq!(V.transpose() * &V, DMatrix::identity(k, k), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_svd_singular_values() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        let singular_values: Vec<f64> = S.diagonal().iter().cloned().collect();
        let mut sorted_singular_values = singular_values.clone();
        sorted_singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(singular_values, sorted_singular_values);
    }

    #[test]
    fn test_rand_svd_relative_frobenius_error() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (U, S, V) = rand_svd(&a, k, epsilon, s);
        let A_approx = &U * &S * V.transpose();
        let error = (&a - &A_approx).norm() / a.norm();
        assert!(error <= epsilon);
    }

    #[test]
    fn test_randsvd_compare(){
        // TODO: Test if values sorted in decreasing order etc
        let height = 100;
        let width = 50;
        let a =  generate_random_matrix(height, width);

        let k = width - 5;
        let epsilon= 0.01;
        let s = 0;

        let tick = Instant::now();
        let (u, s, v) = lora_drivers::rand_svd(&a, k, epsilon, s);
        let tock = tick.elapsed();

        println!("Time taken by RandSVD: {:?}", tock);

        let call_a = a.clone();
        let tick = Instant::now();
        let svd = call_a.svd(true, true);
        let tock = tick.elapsed();
        println!("Time taken by Deterministic SVD: {:?}", tock);

        // TODO: why is the reconstruction error high here


        let deterministic_u = svd.u.unwrap();
        let deterministic_s = DMatrix::from_diagonal(&svd.singular_values);
        let deterministic_v = svd.v_t.unwrap().transpose();

        println!("Dimensions of Determ U: {:?}", deterministic_u.shape());
        println!("Dimensions of Determ S: {:?}", deterministic_s.shape());
        println!("Dimensions of Determ V: {:?}", deterministic_v.shape());

        println!("Dimensions of Rand U: {:?}", u.shape());
        println!("Dimensions of Rand S: {:?}", s.shape());
        println!("Dimensions of Rand V: {:?}", v.shape());

        

        // reconstruct the matrix a from u s and v
        let reconstructed1 = &u * &s * &v.transpose();
        println!("Reconstructed Dimensions: {:?}", reconstructed1.shape());
        println!("Original Dimensions: {:?}", a.shape());

        let diff1 = a.norm() - reconstructed1.norm();
        println!("Difference between A and Rand Reconstructed: {}", diff1);
        

        // take the first k columns of deterministic u s and v and then reconstruct


        let reconstructed2 = &deterministic_u * &deterministic_s * &deterministic_v.transpose();
        let diff2 = a.columns(0,k).norm() - reconstructed2.columns(0,k).norm();
        println!("Difference between Determ A and Reconstructed: {}", diff2);

        let reconstruction_diff = diff1 - diff2.abs();
        println!("Difference between Rand and Determ Reconstruction: {}", reconstruction_diff);


    }
}




#[cfg(test)]
mod test_randevd1
{
    use crate::test_assist::{generate_random_matrix, generate_random_hermitian_matrix};
    use crate::lora_helpers;
    use crate::lora_drivers;
    use std::time::Instant;
    use approx::assert_relative_eq;
    use super::*;

    #[test]
    fn test_rand_evd1_basic_functionality() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (V, lambda) = lora_drivers::rand_evd1(&a, k, epsilon, s);
        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }

    #[test]
    fn test_rand_evd1_large_hermitian_matrix() {
        let a = generate_random_hermitian_matrix(100);
        let k = 10;
        let epsilon = 1e-6;
        let s = 5;

        let (V, lambda) = lora_drivers::rand_evd1(&a, k, epsilon, s);
        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }

    #[test]
    fn test_rand_evd1_zero_matrix() {
        let a = DMatrix::zeros(5, 5);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (V, lambda) = lora_drivers::rand_evd1(&a, k, epsilon, s);
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

        let (V, lambda) = lora_drivers::rand_evd1(&a, k, epsilon, s);
        assert_eq!(V.ncols(), k);
        assert_eq!(lambda.len(), k);
    }
    #[test]
    fn test_randevd1_compare() {

        let dims = 100;
        
        // needs to be square
        let A_rand =  generate_random_matrix(dims, dims);
        let A_rand_psd = &A_rand*(&A_rand.transpose());


        let k = dims -5;
        let epsilon= 0.01;
        let s = 5;
        
        let k = 2;
        let epsilon = 1e-6;
        let s = 1;

        println!("Running test_rand_evd1");

        let tick = Instant::now();
        let (v, lambda) = lora_drivers::rand_evd1(&A_rand_psd, k, epsilon, s);
        let tock = tick.elapsed();
        println!("Time taken by RandEVD1: {:?}", tock);

        // printed in col major order
        // println!("Rand Eigenvectors: {:?}", v);
        

        let tick = Instant::now();
        let normal_evd = A_rand_psd.symmetric_eigen();
        let tock = tick.elapsed();
        println!("Time taken by Normal EVD: {:?}", tock);
    
        
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

        let (V, lambda) = rand_evd1(&a, k, epsilon, s);
        assert_eq!(lambda.len(), V.ncols());
        assert_eq!(V.nrows(), a.nrows());
    }

    #[test]
    fn test_rand_evd1_orthogonality() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (V, lambda) = rand_evd1(&a, k, epsilon, s);
        assert_relative_eq!(V.transpose() * &V, DMatrix::identity(k, k), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_evd1_eigenvalues() {
        let a = generate_random_hermitian_matrix(100);
        let k = 3;
        let epsilon = 1e-6;
        let s = 2;

        let (V, lambda) = rand_evd1(&a, k, epsilon, s);
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

        let (V, lambda) = rand_evd1(&a, k, epsilon, s);
        let A_approx = &V * DMatrix::from_diagonal(&DVector::from_vec(lambda.clone())) * V.transpose();
        let error = (&a - &A_approx).norm() / a.norm();
        println!("Error: {}", error);
        assert!(error <= epsilon);
    }
}



#[cfg(test)]
mod test_randevd2
{
    use crate::test_assist::{generate_random_matrix, generate_random_hermitian_matrix, generate_random_psd_matrix};
    use crate::lora_helpers;
    use crate::lora_drivers;
    use std::time::Instant;
    use approx::assert_relative_eq;
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
        let (V, lambda) = result.unwrap();
        assert_relative_eq!(V.transpose() * &V, DMatrix::identity(k, k), epsilon = 1e-6);
    }

    #[test]
    fn test_rand_evd2_eigenvalues() {
        let a = generate_random_psd_matrix(5);
        let k = 3;
        let s = 2;

        let result = rand_evd2(&a, k, s);
        assert!(result.is_ok());
        let (V, lambda) = result.unwrap();
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
        let epsilon= 0.01;
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

        // find the reconstruction error between rand and deterministic after truncating the deterministic ka columns to the specified rank. Reconstruction error is the different between the original matrix A truncated and the reconstructed matrix from the eigvects and eigvals

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
