use nalgebra::DMatrix;
use crate::sketch::{DistributionType, sketching_operator};
use crate::pivot_decompositions::qrcp;

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
    let r_sk_k_inv = r_sk_k.try_inverse().expect("Failed to compute inverse");
    let a_pre = a_perm * r_sk_k_inv;

    let ata = a_pre.transpose() * &a_pre;
    let chol = ata.cholesky().expect("Cholesky decomposition failed");
    let r_pre = chol.l().transpose();
    let q = a_pre * r_pre.clone().try_inverse().expect("Failed to compute inverse");

    let r = r_pre * r_sk.view((0, 0), (k, n));

    (q, r, j)
}



#[cfg(test)]
mod tests
{
    use nalgebra::DMatrix;
    use crate::cqrrpt::sap_chol_qrcp;
    use crate::pivot_decompositions::qrcp;
    use crate::sketch::{sketching_operator, DistributionType};
    use rand::Rng;
    use std::time::Instant;
    
    fn permutation_vector_to_transpose_matrix(perm: &[usize]) -> DMatrix<f64> {
        let n = perm.len();
        let mut perm_matrix = DMatrix::<f64>::zeros(n, n); // Initialize an n x n matrix with zeros
    
        for (i, &p) in perm.iter().enumerate() {
            perm_matrix[(i, p)] = 1.0;
        }
    
        perm_matrix
    }
    fn check_approx_equal(a: &DMatrix<f64>, b: &DMatrix<f64>, tolerance: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if (a[(i, j)] - b[(i, j)]).abs() > tolerance {
                    println!("{}, {}, {}, {}", i, j, a[(i, j)], b[(i, j)]);
                    return false;
                }
            }
        }
        
        true
    }
    
    fn check_upper_triangular(a: &DMatrix<f64>, tolerance: f64) -> bool {
        
        for i in 0..a.nrows() {
            for j in 0..i.min(a.ncols()) {
                if (a[(i, j)]).abs() > tolerance {
                    // println!("({}, {}), {}", i, j, a[(i, j)]);
                    return false;
                }
            }
        }
        true
    }
    #[test]
    fn test_cqrrpt(){
        let n = 100;
        let m = 1000;
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
        let qtq = &q.transpose()*&q;
        assert!(check_upper_triangular(&r, 1e-4));
        assert!(check_approx_equal(&qtq,  &DMatrix::identity(q_cp.ncols(), q_cp.ncols()), 1e-4));
        assert!(check_approx_equal(&reconstruct, &reconstructed, 1e-4));

        print!("Deterministic Time {:2?}", duration2);
        print!("Sketched Time {:2?}", duration1);
    }
    
}