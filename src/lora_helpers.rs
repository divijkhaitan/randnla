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
    let Y = A*S;
    let Q = Orth(&Y);
    return Q; 
}

// Range Finder
// ======================================================================================



// ======================================================================================
// Tall Sketch Operator Generator







// Tall Sketch Operator Generator
// ======================================================================================










// ======================================================================================
// Orth methods

pub fn Orth(X: &DMatrix<f64>) -> DMatrix<f64> {
    X.clone().qr().q()
}

pub struct Stabilizer {
}

impl Stabilizer {
    pub fn new() -> Self {
        Stabilizer {}
    }

    
    pub fn stabilize(&self, y: &mut DMatrix<f64>) {
        let qr = y.clone().qr();
        *y = qr.q(); // we just need to output the q factor from the QR decomposition
    }
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
    use rand_123::rng::ThreeFry2x64Rng;
    use rand_core::{SeedableRng, RngCore};
    use rand::Rng;
    use std::time::Instant;
    use rand_distr::{Distribution, Normal, Uniform, Bernoulli, StandardNormal};
    use rand::distributions::DistIter;


}




