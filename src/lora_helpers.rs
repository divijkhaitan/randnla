use nalgebra::{DMatrix, dmatrix, ColPivQR};
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
// Tall Sketch Operator Generator







// Tall Sketch Operator Generator
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




// =================================================================================================
// For CURD1


pub fn osid_qrcp(Y: &DMatrix<f64>, k: usize, axis: usize) -> (DMatrix<f64>, Vec<usize>) {
    // axis == 2 means columnID which is what we need
    if axis == 2 {
        let (l, w) = (Y.nrows(), Y.ncols());
        assert!(k <= l && k <= w);

        // TODO: see economic
        let col_piv_qr = Y.clone().col_piv_qr();
        let R = col_piv_qr.r();
        let J = col_piv_qr.p();

        // needed to add this functionality to nalgebra
        let J_indices = J.permutation_indices();
 

        let Q = col_piv_qr.q();

        // from DOCS: .view(start, shape):  Reference to the submatrix with shape.0 rows and shape.1 columns, starting with the start.0-th row and start.1-th column. start and shape are both tuples.
        let R1 = R.view((0,0), (k,k));
        let R2 = R.view((0,k), (k, w-k));
        // the solve method wasn't found for the matrices so couldnt do R1.solve(R2)

        let T_matrix = R1.try_inverse().expect("R11 is not invertible") * (&R2);

        let mut X = DMatrix::<f64>::zeros(k, w);


        let J_indices = J.permutation_indices(); // Vec<(usize, usize)>
        

        let mut permutation_indices: Vec<usize> = (0..w).collect();
        
        // get final permutation
        for &(i1, i2) in J_indices.iter() {
            permutation_indices.swap(i1, i2);
        }

        // need manual approach for simple indexing and horizontal stacking
        for i in 0..k {  
            for j in 0..w {  
                let j_idx = permutation_indices[j];  // Get permuted column index
                if j < k {
                    
                    X[(i, j_idx)] = if i == j { 1.0 } else { 0.0 };
                } else {
                    
                    X[(i, j_idx)] = T_matrix[(i, j-k)];
                }
            }
        }

        let J: Vec<usize> = permutation_indices.iter().take(k).cloned().collect();

        return (X, J)

    } else {
        // For axis == 1 (row ID), transpose and call with axis = 2
        let Y_trans = Y.transpose();
        let (X, I) = osid_qrcp(&Y_trans, k, 2);
        return (X.transpose(), I)
    }

}





// For CURD1
// =================================================================================================

// =================================================================================================
// Misc



// =================================================================================================











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

    

    #[test]
    fn test_osid_qrcp(){
        let a = dmatrix![1.0, 2.0, 3.0;
                    4.0, 5.0, 6.0;
                    7.0, 8.0, 9.0];
        
        let k = 2;
        let axis = 2;
        let (X,Js) = lora_helpers::osid_qrcp(&a, k, axis);
        println!("X: \n{}", X);
        println!("Js: \n{:?}", Js);


    }


}




