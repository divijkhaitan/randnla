use nalgebra::DMatrix;
use crate::sketch::{DistributionType, sketching_operator};
use crate::pivot_decompositions::qrcp;

// Sketch and Precondition for Cholesky QR with Column Pivoting  
/**
Computes a column pivoted QR factorisation by preconditioning and using CholeskyQR
* Inputs:
a: m x n matrix (m >> n)  
d: integer between n and m  

* Output:  
A tuple (q, r, p) containing:  
q: An orthogonal matrix of size m x m  
r: An upper triangular matrix of size m x n  
p: A vector indicating the permutation of columns applied during pivoting  

* Notes:  
Can be used as a subroutine for HQRRP to extend to general matrices  
Panics for matrices that are not tall and invalid values of d  

Computes the QR Decomposition of a matrix using the cholesky QR  
method. It computes the QR decomposition of a sketch of A, uses the triangular  
factor as a preconditioner and returns the upper trigangular factor from the  
cholesky decomposition of (A_pre)*A_pre  
*/
pub fn sap_chol_qrcp(a: &DMatrix<f64>, d: usize) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let (m, n) = a.shape();
    assert!(n <= d && d <= m, "d must satisfy n ≤ d ≪ m");

    let s = sketching_operator(DistributionType::Gaussian, d, m).unwrap();

    let sa = &s * a;
    
    let (_, r_sk, j) = qrcp(&sa);
    
    let tol = 1e-10;
    let mut k = 0;
    for i in 0..r_sk.nrows().min(r_sk.ncols()) {
        if r_sk[(i, i)].abs() > tol {
            k += 1;
        }
    }

    let r_sk_k = r_sk.view((0, 0), (k, k));
    let a_perm = a.select_columns(&j[..k]);
    let r_sk_k_inv = r_sk_k.solve_upper_triangular(&DMatrix::identity(k, k)).unwrap();
    let a_pre = a_perm * r_sk_k_inv;

    let ata = a_pre.transpose() * &a_pre;
    let chol = ata.cholesky().expect("Cholesky decomposition failed");
    let r_pre = chol.l().transpose();
    let q = a_pre * r_pre.solve_upper_triangular(&DMatrix::identity(n, n)).unwrap();

    let r = r_pre * r_sk.view((0, 0), (k, n));

    (q, r, j)
}



#[cfg(test)]
mod tests
{
    use approx::assert_relative_eq;
    use crate::cqrrpt::sap_chol_qrcp;
    use crate::pivot_decompositions::qrcp;
    use crate::sketch::{sketching_operator, DistributionType};
    use rand::Rng;
    use std::time::Instant;
    use crate::test_assist::{check_approx_equal, permutation_vector_to_transpose_matrix, check_upper_triangular};
    #[test]
    fn test_cqrrpt(){
        let n = 10;
        let m = 100;
        let d = rand::thread_rng().gen_range(n..m);
        
        let data = sketching_operator(DistributionType::Uniform, m, n).unwrap();
        
        let start1 = Instant::now();
        let (q, r, j) = sap_chol_qrcp(&data, d);
        let duration1 = start1.elapsed();

        let start2 = Instant::now();
        let (q_cp, r_cp, p) = qrcp(&data);
        let p_cp = permutation_vector_to_transpose_matrix(&p);
        let duration2 = start2.elapsed();
        

        let reconstruct = &q*&r*permutation_vector_to_transpose_matrix(&j);
        let reconstructed = &q_cp*&r_cp*&p_cp;
        println!("Deterministic Time {:2?}", duration2);
        println!("Sketched Time {:2?}", duration1);        
        
        // Normal columns
        let cols = q.ncols();
        for j in 0..cols {
            assert_relative_eq!(q.column(j).norm(), 1.0, epsilon = 1e-6);
        }

        // Orthogonal columns
        for i in 0..cols {
            for j in (i+1)..cols {
                assert_relative_eq!(q.column(i).dot(&q.column(j)), 0.0, epsilon = 1e-6);
            }
        }

        assert!(check_upper_triangular(&r, 1e-4));
        assert!(check_approx_equal(&reconstruct, &reconstructed, 1e-4));

    }
    #[test]
    #[should_panic(expected = "d must satisfy n ≤ d ≪ m")] // Optional: specify panic message
    fn test_cqrrpt_wrong_d()
    {
        let n = 10;
        let m = 100;
        let d = rand::thread_rng().gen_range(0..n);
        
        let data = sketching_operator(DistributionType::Uniform, m, n).unwrap();
        sap_chol_qrcp(&data, d);
    }
    #[test]
    #[should_panic(expected = "d must satisfy n ≤ d ≪ m")] // Optional: specify panic message
    fn test_cqrrpt_wide_matrix()
    {
        let n = 10;
        let m = 100;
        let d = rand::thread_rng().gen_range(n..m);
        
        let data = sketching_operator(DistributionType::Uniform, n, m).unwrap();
        sap_chol_qrcp(&data, d);
    }
}