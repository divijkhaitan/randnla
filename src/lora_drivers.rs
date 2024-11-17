use nalgebra::{DMatrix, dmatrix};
use crate::sketch;
use crate::lora_helpers;


/*
TODO: Remove the use of clones and try to optimize performance wherever you can
 */



// randomized SVD
pub fn rand_SVD(A:&DMatrix<f64>, k:usize, epsilon: f64, s: usize) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)
{
    println!("Running RSVD");
    
    // TODO: check the positive and negative switching of the values compared to Python
    let (Q,B) = lora_helpers::QB1(&A, k+s, epsilon);
    let r = std::cmp::min(k, Q.ncols());
    let mysvd_rand = B.clone().svd(true, true);
    let U_binding = mysvd_rand.u.unwrap();
    let U = U_binding.columns(0, r).clone();

    let V_binding = mysvd_rand.v_t.unwrap().transpose();
    let V = V_binding.columns(0, r).clone();

    let S_binding = DMatrix::from_diagonal(&mysvd_rand.singular_values);
    let S = S_binding.rows(0, r).clone();
    let U_final = Q*U;
    return (U_final, S.into(), V.into());

}

    
// only for positive semi-definite matrices
// For performance reasons we are not checking if the matrix is symmetric since that would lead to a performance hit
pub fn rand_EVD2(A:&DMatrix<f64>, k:usize, s: usize) -> Result<(DMatrix<f64>, Vec<f64>), &'static str> {
    println!("Running REVD2");
    let S = sketch::sketching_operator(sketch::DistributionType::Gaussian, A.nrows(), k+s);
    let Y = A*&S;
    let mach_eps = f64::EPSILON;
    let nu = (A.nrows() as f64).sqrt() * mach_eps * Y.norm();
    let Y_new = Y + (nu*&S);
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
    fn test_randSVD(){
        let a = dmatrix![1.0, 2.0, 3.0;
                    4.0, 5.0, 6.0;
                    7.0, 8.0, 9.0];

        let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dims = 10;
        let a =  DMatrix::from_fn(dims, dims, |_i, _j| normal.sample(&mut rng_threefry));

        let k = dims;
        let epsilon= 0.01;
        let s = 5;

        let tick = Instant::now();
        let (u, s, v) = lora_drivers::rand_SVD(&a, k, epsilon, s);
        let tock = tick.elapsed();

        println!("Time taken by RandSVD: {:?}", tock);

        let tick = Instant::now();
        let svd = a.svd(true, true);
        let tock = tick.elapsed();
        println!("Time taken by Deterministic SVD: {:?}", tock);


        let deterministic_u = svd.u.unwrap();
        let deterministic_s = DMatrix::from_diagonal(&svd.singular_values);
        let deterministic_v = svd.v_t.unwrap().transpose();


        let diff_u = (&u - &deterministic_u).norm();
        let diff_s = (&s - &deterministic_s).norm();
        let diff_v = (&v - &deterministic_v).norm();
        println!("Difference between U: {}", diff_u);
        println!("Difference between S: {}", diff_s);
        println!("Difference between V: {}", diff_v);

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
        let randevd2 = lora_drivers::rand_EVD2(&A_rand_psd, k, s);
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


