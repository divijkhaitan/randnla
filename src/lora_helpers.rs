use nalgebra::{DMatrix};
use rand::Rng;
use crate::sketch;







// ===================================================================================================
// QB Decomposer

//     /*
//     Q = RangeFinder(A, k, epsilon)
//     B = Q^*A
//     return Q, B
//      */
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
// ===================================================================================================


// ===================================================================================================
// Range Finder

//     /*
//     S = TallSketchOpGen(A, k)
//     Y = AS
//     Q = orth(Y)
//     return Q
//     */
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
// ===================================================================================================










// ===================================================================================================
// Orth methods

pub fn Orth(X: &DMatrix<f64>) -> DMatrix<f64> {
    X.clone().qr().q()
}






pub struct Stabilizer {
    // Parameters for stabilization method can be added here
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
// =================================================================================================








