#![allow(dead_code)]
#![allow(warnings)]
#![allow(unused_imports)]

use nalgebra::{DMatrix, dmatrix, ColPivQR};
use rand::Rng;
use crate::sketch;







// ======================================================================================
// QB Decomposer

pub fn QB1(A: &DMatrix<f64>, k: usize, epsilon: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let Q = RF1(A, k);
    // println!("Q: \n{}", Q);
    // println!("A: \n{}", A);
    let B = Q.clone().transpose()*A;
    return (Q, B);
}




// pub fn QB2(A: &DMatrix<f64>, k: usize, epsilon: f64) -> (DMatrix<f64>, DMatrix<f64>) {

//     let mut q = DMatrix::from_element(m, k + s, 0.0);
//     let mut b = DMatrix::from_element(k + s, n, 0.0);

//     (q, b)
// }


// QB Decomposer
// ======================================================================================


// ======================================================================================
// Range Finder

// does this need to take epsilon parameter?
pub fn RF1(A: &DMatrix<f64>, k: usize) -> DMatrix<f64> {
    let n = A.nrows();
    // replace with TallSketchOpGen
    let S = sketch::sketching_operator(sketch::DistributionType::Gaussian, n, k);
    let Y = A*&(S.unwrap());
    let Q = Orth(&Y);
    return Q; 
}

// Range Finder
// ======================================================================================



// ======================================================================================
// Tall Sketch Operator Generator

pub fn tsog1(A: &DMatrix<f64>, k: usize, s: usize, num_passes: i32, passes_per_stab: i32 ) -> DMatrix<f64> {

    let mut passes_done = 0;
    let mut S: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::from_element(1, 1, 0.0);
    if num_passes % 2 == 0 {
        S = sketch::sketching_operator(sketch::DistributionType::Gaussian, A.ncols(), k).unwrap();
    }
    else {
        let pre_sketch = sketch::sketching_operator(sketch::DistributionType::Gaussian, A.nrows(), k).unwrap();

        let mut S1 = A.transpose()*pre_sketch;
        passes_done += 1;
        if (passes_done % passes_per_stab == 0)
        {
            S = Stabilizer(&S1);
        } 
    }

    let mut diff = num_passes - passes_done;

    while diff >= 2 {
        let S1 = A*&S;
        passes_done += 1;
        if (passes_done % passes_per_stab == 0)
        {
            S = Stabilizer(&S1);
        } 
        let S2 = A.transpose()*S1;
        passes_done += 1;
        if (passes_done % passes_per_stab == 0)
        {
            S = Stabilizer(&S2);
        } 
        diff -= 2;
    }

    return S;
}






// Tall Sketch Operator Generator
// ======================================================================================










// ======================================================================================
// Orth methods

pub fn Orth(X: &DMatrix<f64>) -> DMatrix<f64> {
    X.clone().qr().q()
}


pub fn Stabilizer(X: & DMatrix<f64>) -> DMatrix<f64> {
    X.clone().qr().q()
}

// Orth methods
// ====================================================================================






// ====================================================================================
// Misc



// ====================================================================================











#[cfg(test)]
mod test_helpers
{
    use nalgebra::{DMatrix, DVector, dmatrix, dvector};
    use crate::lora_helpers;
    use crate::lora_drivers;
    use crate::sketch;
    use rand_123::rng::ThreeFry2x64Rng;
    use rand_core::{SeedableRng, RngCore};
    use rand::Rng;
    use std::time::Instant;
    use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};
    use rand::distributions::DistIter;


    #[test]
    fn test_tsog1_sketch() {

        let mut rng_threefry = ThreeFry2x64Rng::seed_from_u64(0);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dims = 10;
        let a =  DMatrix::from_fn(dims, dims, |_i, _j| normal.sample(&mut rng_threefry));
        let k = 5;
        let s = 0;
        let num_passes = 1;
        let passes_per_stab = 1;

        let S = lora_helpers::tsog1(&a, k, s, num_passes, passes_per_stab);
        println!("S: \n{}", S);
    }


}




