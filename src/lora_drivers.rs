use nalgebra::{DMatrix};
use crate::lora_helpers;







// randomized SVD
pub fn rand_SVD(A:&DMatrix<f64>, k:usize, epsilon: f64, s: usize) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)
{
    println!("Running RSVD");
    /*
    Q, B = QBDecomposer(A, k+s, epsilon)
    r = min(k, Q.ncols())
    U, S, V = svd(B)
    U = U[:, :r]
    V = V[: , : r]
    S = S[: r , : r]
    U = QU
    return U, S, V
     */
    
    // TODO: check the positive and negative switching of the values
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
    fn test_drivers(){
        let a = dmatrix![1.0, 2.0, 3.0;
                    4.0, 5.0, 6.0;
                    7.0, 8.0, 9.0];

        let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dims = 10;
        let a =  DMatrix::from_fn(dims, dims, |_i, _j| normal.sample(&mut rng_threefry));
        // println!("A: \n{}", a);

        let k = dims;
        let epsilon= 0.01;
        let s = 5;

        let tick = Instant::now();
        let (u, s, v) = lora_drivers::rand_SVD(&a, k, epsilon, s);
        let tock = tick.elapsed();

        println!("Time taken by RandSVD: {:?}", tock);
        // println!("U: \n{}", u);
        // println!("S: \n{}", s);
        // println!("V: \n{}", v);

        // normal nalgebra svd

        let tick = Instant::now();
        let svd = a.svd(true, true);
        let tock = tick.elapsed();
        println!("Time taken by Deterministic SVD: {:?}", tock);


        let deterministic_u = svd.u.unwrap();
        let deterministic_s = DMatrix::from_diagonal(&svd.singular_values);
        let deterministic_v = svd.v_t.unwrap().transpose();
        // println!("Deterministic U: \n{}", deterministic_u);
        // println!("Deterministic S: \n{}", deterministic_s);
        // println!("Deterministic V: \n{}", deterministic_v);

        let diff_u = (&u - &deterministic_u).norm();
        let diff_s = (&s - &deterministic_s).norm();
        let diff_v = (&v - &deterministic_v).norm();
        println!("Difference between U: {}", diff_u);
        println!("Difference between S: {}", diff_s);
        println!("Difference between V: {}", diff_v);





    }

}


