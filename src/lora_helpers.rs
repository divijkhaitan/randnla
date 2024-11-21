use nalgebra::DMatrix;
use crate::sketch;

/**
Takes an `m × n` matrix and `k << min{m, n}` a positive integer and returns `Q` an `m × d` matrix returned by the underlying RangeFinder and `B = Q*A` an `d × n` matrix

* Inputs:
`A` is an `m × n` matrix and `k << min{m, n}` is a positive integer.

* Output:
`Q` an `m × d` matrix returned by the underlying RangeFinder and
`B = Q*A` is a `d × n` matrix
`QB` is a low-rank approximation of `A`.
(<https://arxiv.org/pdf/2302.11474>)
 */

 pub fn QB1(A: &DMatrix<f64>, k: usize, epsilon: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let _ = epsilon;
    // we don't control the approximation error with this implementation of QB1, so epsilon is ignored
    let Q = RF1(A, k);
    let B = Q.transpose() * A;
    return (Q, B)
}

/**
Takes an `m x n` matrix and `0 < k << min{m, n}` and returns `Q` a column orthonormal matrix with `d = min{k, rank A}` columns

* Input:
`A` is `m × n`, and `k << min{m, n}` is a positive integer

* Output:
`Q` is a column-orthonormal matrix with `d = min{k, rank A}` columns


The range of Q is an approximation to the space spanned by A’s top d left singular vectors.
 */
pub fn RF1(A: &DMatrix<f64>, k: usize) -> DMatrix<f64> {

    // 2 and 1 are numbers for num_passes and passes_per_stab as advised in the monograph and other code
    let sketch_S = tsog1(A, k, 2, 1);
    let Y = A*&(sketch_S);
    let Q = Orth(&Y);
    return Q; 
}



/** 
Data-aware sketching based on power iteration. Take an `m × n` matrix and `0 < k << min{m, n}` and returns an `n × k` sketch `S` 

 * Input:
 `A` is `m × n`, and `k << min{m, n}` is a positive integer
 `num_passes` controls the number of passes
 `passes_per_stab` controls the frequency of stabilizer computation (number of matmuls with A or A* before stabilizer is called)
* Output:
`S` is `n × k`, intended for later use in computing `Y = A*S`
 */
pub fn tsog1(A: &DMatrix<f64>, k: usize, num_passes: i32, passes_per_stab: i32 ) -> DMatrix<f64> {

    let mut passes_done = 0;
    let n = A.ncols();
    
    let mut S1: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::from_element(n, k, 0.0);

    let mut _S2: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::from_element(n, k, 0.0);

    let mut S: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::from_element(n, k, 0.0);


    if num_passes % 2 == 0 {
        S = sketch::sketching_operator(sketch::DistributionType::Gaussian, A.ncols(), k).unwrap();
    }
    else {
        S1 = sketch::sketching_operator(sketch::DistributionType::Gaussian, A.nrows(), k).unwrap();

        S1 = A.transpose()*S1;
        passes_done += 1;
        if passes_done % passes_per_stab == 0
        {
            _S2 = Stabilizer(&S1);
        } 
    }

    
    let mut diff = num_passes - passes_done;
    
    
    while diff >= 2 {
        S = A*&S1;
        passes_done += 1;
        if passes_done % passes_per_stab == 0
        {
            S = Stabilizer(&S);
        } 
        S = A.transpose()*S;
        passes_done += 1;
        if passes_done % passes_per_stab == 0
        {
            S = Stabilizer(&S);
        } 
        diff -= 2;
    }

    return S;
}









// Orth methods
// needs to use economic qr decomposition
// returns orthogonal Q factor from a QR decomposition
/**
 Returns orthonormal basis for the range of X

 * Input:
`X` is `m × n` matrix

 * Output:
`Y` is an orthonormal basis for the range of X

The number of columns in Y will never be larger than that of X and may be
smaller. We have used the simplest method and returned the orthogonal
factor from a QR decomposition of X.
 */
pub fn Orth(X: &DMatrix<f64>) -> DMatrix<f64> {
    X.clone().qr().q()
}


/**
Less expensive alternative to Orth(X) that returns a lower-triangular matrix L from LU decomp. Only requires Y to be better-conditioned than X while preserving its range.

 * Input:
 `X` is `m × n` matrix
 * Output:
`Y` is lower triangular matrix from LU decomposition of X with column pivoting
 */
pub fn Stabilizer(X: & DMatrix<f64>) -> DMatrix<f64> {
    X.clone().full_piv_lu().l()
}



#[cfg(test)]
mod test_other_helpers
{
    use super::*;
    use crate::test_assist::generate_random_matrix;
    use nalgebra::DMatrix;
    use crate::lora_helpers;

    // TODO: How to test/benchmark this for sketch quality??
    #[test]
    fn test_tsog1_sketch() {
        let dims = 5;
        let a =  generate_random_matrix(dims, dims);
        let k = 3;
        let num_passes = 3;
        let passes_per_stab = 1;

        let S = lora_helpers::tsog1(&a, k, num_passes, passes_per_stab);
        assert!(S.nrows() == dims);
        assert!(S.ncols() == k);
    }

    #[test]
    fn test_tsog1_basic_functionality() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let num_passes = 4;
        let passes_per_stab = 2;

        let S = tsog1(&a, k, num_passes, passes_per_stab);
        assert_eq!(S.ncols(), k);
    }

    #[test]
    fn test_tsog1_even_num_passes() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let num_passes = 4;
        let passes_per_stab = 2;

        let S = tsog1(&a, k, num_passes, passes_per_stab);
        assert_eq!(S.ncols(), k);
    }

    #[test]
    fn test_tsog1_odd_num_passes() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let num_passes = 3;
        let passes_per_stab = 2;

        let S = tsog1(&a, k, num_passes, passes_per_stab);
        assert_eq!(S.ncols(), k);
    }

    #[test]
    fn test_tsog1_different_stabilizer_frequencies() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let num_passes = 4;

        let passes_per_stab = 1;
        let S1 = tsog1(&a, k, num_passes, passes_per_stab);
        assert_eq!(S1.ncols(), k);

        let passes_per_stab = 2;
        let S2 = tsog1(&a, k, num_passes, passes_per_stab);
        assert_eq!(S2.ncols(), k);

        let passes_per_stab = 3;
        let S3 = tsog1(&a, k, num_passes, passes_per_stab);
        assert_eq!(S3.ncols(), k);
    }

    #[test]
    fn test_tsog1_identity_matrix() {
        let a = DMatrix::identity(5, 5);
        let k = 3;
        let num_passes = 4;
        let passes_per_stab = 2;

        let S = tsog1(&a, k, num_passes, passes_per_stab);
        assert_eq!(S.ncols(), k);
    }



    // ========= Range Finder Tests =========
    #[test]
    fn test_rf1_basic_functionality() {
        let a = generate_random_matrix(100, 50);
        let k = 3;

        let Q = RF1(&a, k);
        assert_eq!(Q.ncols(), k);
    }

    #[test]
    fn test_rf1_accuracy() {
        let a = generate_random_matrix(100, 50);
        let k = 3;

        let Q = RF1(&a, k);
        let error = (a.clone() - (Q.clone() * (Q.transpose() * a.clone()))).norm();
        assert!(error < a.norm());
    }

    // ========= Range Finder Tests =========


    // ========= QB Tests =========

    #[test]
    fn test_qb1_basic_functionality() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1e-6;

        let (Q, B) = QB1(&a, k, epsilon);
        assert_eq!(Q.ncols(), k);
        assert_eq!(B.nrows(), k);
        assert_eq!(B.ncols(), a.ncols());
    }

    #[test]
    fn test_qb1_accuracy_check() {
        let a = generate_random_matrix(10, 5);
        let k = 3;
        let epsilon = 1.0;

        let (Q, B) = QB1(&a, k, epsilon);
        let approx = &Q * &B;
        let error = (&a - &approx).norm() / a.norm();
        assert!(error <= epsilon);
    }


    // ========= QB Tests =========



}





#[cfg(test)]
mod test_lower_helpers
{
    use super::*;
    use crate::test_assist::generate_random_matrix;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;


    #[test]
    fn test_orth_random_tall_matrix() {
        let a = generate_random_matrix(10, 5);
        let q = Orth(&a);
        assert_eq!(q.nrows(), 10);
        assert!(q.ncols() <= 5);
        assert_relative_eq!(q.transpose() * &q, DMatrix::identity(q.ncols(), q.ncols()), epsilon = 1e-6);
    }

    #[test]
    fn test_orth_square_matrix() {
        let a = generate_random_matrix(5, 5);
        let q = Orth(&a);
        assert_eq!(q.nrows(), 5);
        assert_eq!(q.ncols(), 5);
        assert_relative_eq!(q.transpose() * &q, DMatrix::identity(5, 5), epsilon = 1e-6);
    }

    #[test]
    fn test_orth_zero_matrix() {
        let a = DMatrix::zeros(5, 5);
        let q = Orth(&a);
        assert_eq!(q.nrows(), 5);
        assert_eq!(q.ncols(), 5);
        assert_relative_eq!(q, DMatrix::identity(5, 5), epsilon = 1e-6);
    }

    #[test]
    fn test_orth_identity_matrix() {
        let a = DMatrix::identity(5, 5);
        let q = Orth(&a);
        assert_eq!(q.nrows(), 5);
        assert_eq!(q.ncols(), 5);
        assert_relative_eq!(q, DMatrix::identity(5, 5), epsilon = 1e-6);
    }

    #[test]
    fn test_stabilizer_random_tall_matrix() {
        let a = generate_random_matrix(10, 5);
        let l = Stabilizer(&a);
        assert_eq!(l.nrows(), 10);
        assert!(l.ncols() <= 5);
    }

    #[test]
    fn test_stabilizer_square_matrix() {
        let a = generate_random_matrix(5, 5);
        let l = Stabilizer(&a);
        assert_eq!(l.nrows(), 5);
        assert_eq!(l.ncols(), 5);
    }

    #[test]
    fn test_stabilizer_zero_matrix() {
        let a = DMatrix::zeros(5, 5);
        let l = Stabilizer(&a);
        assert_eq!(l.nrows(), 5);
        assert_eq!(l.ncols(), 5);
        assert_relative_eq!(l, DMatrix::identity(5, 5), epsilon = 1e-6);
    }

    #[test]
    fn test_stabilizer_identity_matrix() {
        let a = DMatrix::identity(5, 5);
        let l = Stabilizer(&a);
        assert_eq!(l.nrows(), 5);
        assert_eq!(l.ncols(), 5);
    }

}






