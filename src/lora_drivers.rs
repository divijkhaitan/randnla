#![allow(dead_code)]
#![allow(warnings)]
#![allow(unused_imports)]
use nalgebra::{DMatrix, dmatrix};
use crate::sketch;
use crate::lora_helpers;


/*
TODO: Remove the use of clones and try to optimize performance wherever you can
 */



// randomized SVD
/*
A: m x n matrix
The returned approximation will have rank at most k
The approximation produced by the randomized phase of the
algorithm will attempt to A to within eps error, but will not produce
an approximation of rank greater than k + s.
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
    
    (U_final, S, V)
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

pub fn rand_evd1(A: &DMatrix<f64>, k: usize, epsilon: f64, s: usize) -> (DMatrix<f64>, Vec<f64>) {

    /*
    A is nxn Hermitian matrix
     */
    println!("Running REVD1");
    assert!(A == &A.adjoint());
    
    let (Q, B) = lora_helpers::QB1(A, k + s, epsilon);
    
    let C = &B * &Q;
    
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
pub fn rand_evd2(A:&DMatrix<f64>, k:usize, s: usize) -> Result<(DMatrix<f64>, Vec<f64>), &'static str> {
    println!("Running REVD2");
    let S_wrapped = sketch::sketching_operator(sketch::DistributionType::Gaussian, A.nrows(), k+s);
    let S = S_wrapped.unwrap();
    let Y = A*&(S);
    let mach_eps = f64::EPSILON;
    let nu = (A.nrows() as f64).sqrt() * mach_eps * Y.norm();
    let Y_new = Y + (nu*&(S));
    let SY = S.adjoint()*&Y_new;

    let chol = match SY.cholesky() {
        Some(c) => c,
        None => return Err("My Cholesky Failed"),
    };

    // in the monograph, we need the upper triangular part but in nalgebra we get the lower triangular part, so we can't use it directly
    let R = chol.l().transpose();
    let B = Y_new*(R.try_inverse().unwrap());
    let mysvd = B.clone().svd(true, true);

    let V_binding = mysvd.u.unwrap();
    let W_binding = mysvd.v_t.unwrap().transpose();
    let S_binding = DMatrix::from_diagonal(&mysvd.singular_values);
    let lambda = S_binding.iter().filter(|&&x| x > 0.0).map(|x| x*x).collect::<Vec<f64>>();

    // find number of entries in lambda that are greater than nu
    let r = std::cmp::min(k, lambda.iter().filter(|&&x| x > nu).count());
    let lambda1 = lambda.iter().take(r).map(|x| x - nu).collect::<Vec<f64>>();
    let V = V_binding.columns(0, r).clone();

    let V_final = V.into();

    return Ok((V_final, lambda1));
}



#[cfg(test)]
mod test_drivers
{
    use crate::test_assist::generate_random_matrix;
    use nalgebra::{DMatrix, DVector, dmatrix, dvector};
    use crate::lora_helpers;
    use crate::lora_drivers;
    use rand_123::rng::ThreeFry2x64Rng;
    use rand_core::{SeedableRng, RngCore};
    use rand::Rng;
    use std::time::Instant;
    use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};
    use rand::distributions::DistIter;

    
    #[test]
    fn test_randsvd(){

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

        // println!("Dimensions of U: {}", u);
        // println!("Dimensions of S: {}", s);
        // println!("Dimensions of V: {}", v);

        

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

        let reconstruction_diff = (reconstructed1 - reconstructed2).norm();
        println!("Difference between Rand and Determ Reconstruction: {}", reconstruction_diff);


    }


    #[test]
    fn test_rand_evd1() {
        let a = dmatrix![
            2.0, -1.0, 0.0;
            -1.0, 2.0, -1.0;
            0.0, -1.0, 2.0
        ];

        let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dims = 10;
        
        let A_rand =  DMatrix::from_fn(dims, dims, |_i, _j| normal.sample(&mut rng_threefry));
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

    }



    #[test]
    fn test_randEVD2(){
        let A_psd = dmatrix![2.0, -1.0, 0.0;
        -1.0, 2.0, -1.0;
        0.0, -1.0, 2.0];

        let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dims = 10;
        
        let A_rand =  DMatrix::from_fn(dims, dims, |_i, _j| normal.sample(&mut rng_threefry));
        let A_rand_psd = &A_rand*(&A_rand.transpose());


        let k = dims-5;
        let epsilon= 0.01;
        let s = 5;

        let tick = Instant::now();
        let randevd2 = lora_drivers::rand_evd2(&A_rand_psd, k, s);
        let tock = tick.elapsed();

        println!("Time taken by RandEVD2: {:?}", tock);

        let (v_rand, lambda_rand) = match randevd2 {
            Ok((v, lambda)) => {
                println!("OK");
                (v, lambda)
            },
            Err(e) => {
                println!("Error: {}", e);
                return;
            },
        };

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

    }

}


