use nalgebra::{DMatrix, DVector};
use std::error::Error;
use crate::errors::RandNLAError;

// Row Pivoted LU
/*
Performs LU decomposition with partial row pivoting.

* Inputs:
matrix: A square n x n matrix to decompose.

* Output:
A tuple (l, u, p) containing:
l: A lower triangular matrix.
u: An upper triangular matrix.
p: A vector indicating the permutation of rows applied during pivoting.

This algorithm uses partial pivoting to preserve numerical stability during Gaussian elimination.
Returns an error if the matrix is not square or if it is singular.
*/
pub fn lupp(matrix: &DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>, Vec<usize>), Box <dyn Error>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(Box::new(RandNLAError::NotSquare(
            format!("Matrix must be square, found matrix with {} rows and {} columns", matrix.nrows(), matrix.ncols())
        )));
    }

    let mut lu = matrix.clone();
    let mut p: Vec<usize> = (0..n).collect();

    for k in 0..n-1 {
        let mut pivot_row = k;
        let mut pivot_val = lu[(k, k)].abs();

        for i in k+1..n {
            let val = lu[(i, k)].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = i;
            }
        }

        if pivot_val == 0.0 {
            return Err(Box::new(RandNLAError::SingularMatrix(
                format!("Matrix must be nonsingular for an LU decomposition")
            )));
        }

        if pivot_row != k {
            for j in 0..n {
                let temp = lu[(k, j)];
                lu[(k, j)] = lu[(pivot_row, j)];
                lu[(pivot_row, j)] = temp;
            }
            let temp = p[k];
            p[k] = p[pivot_row];
            p[pivot_row] = temp;
        }

        for i in k+1..n {
            let multiplier = lu[(i, k)] / lu[(k, k)];
            lu[(i, k)] = multiplier;  // Store multiplier in L part
            
            for j in k+1..n {
                lu[(i, j)] -= multiplier * lu[(k, j)];
            }
        }
    }

    // Separate L and U matrices
    let mut l = DMatrix::identity(n, n);
    let mut u = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            if i > j {
                l[(i, j)] = lu[(i, j)];
            } else {
                u[(i, j)] = lu[(i, j)];
            }
        }
    }

    Ok((l, u, p))
}

// Column-Pivoted QR
/*
Performs QR decomposition with column pivoting.

* Inputs:
a: An m x n matrix to decompose.

* Output:
A tuple (q, r, p) containing:
q: An orthogonal matrix of size m x m.
r: An upper triangular matrix of size m x n.
p: A vector indicating the permutation of columns applied during pivoting.

This algorithm uses Householder reflections to compute the decomposition
while preserving stability and ordering the diagonal entries of R to
make the decomposition rank-revealing.
*/

pub fn qrcp(
    a: &DMatrix<f64>
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let (m, n) = a.shape();
    let mut q = DMatrix::identity(m, m);
    let mut r = a.clone();
    let mut p: Vec<usize> = (0..n).collect();
    
    let mut col_norms: Vec<f64> = (0..n)
        .map(|j| r.column(j).norm())
        .collect();
    
    for k in 0..n.min(m) {
        let mut max_norm = col_norms[k];
        let mut max_idx = k;
        
        for j in (k + 1)..n {
            if col_norms[j] > max_norm {
                max_norm = col_norms[j];
                max_idx = j;
            }
        }
        
        // Pivot
        if max_idx != k {
            r.swap_columns(k, max_idx);
            p.swap(k, max_idx);
            col_norms.swap(k, max_idx);
        }
        
        // Compute Householder
        let mut x = DVector::zeros(m - k);
        for i in k..m {
            x[i - k] = r[(i, k)];
        }
        
        let norm_x = x.norm();
        if !(norm_x == 0.0) {
            let mut v = x;
            v[0] += if v[0] >= 0.0 { norm_x } else { -norm_x };
            v /= v.norm();
            
            // Reflect R
            for j in k..n {
                let dot_product = v.dot(&r.view((k, j), (m - k, 1)));
                for i in k..m {
                    r[(i, j)] -= 2.0 * v[i - k] * dot_product;
                }
            }
            
            // Update Q
            let h = DMatrix::identity(m, m) - 
                2.0* 
                DMatrix::from_fn(m, m, |i, j| {
                    if i >= k && j >= k {
                        v[i - k] * v[j - k]
                    } else {
                        0.0
                    }
                });
            q = q * h;
            
            for j in (k + 1)..n {
                col_norms[j] = r.view((k + 1, j), (m - k - 1, 1)).norm();
            }
        }
    }
    
    (q, r, p)
}

// Economic QR
/**
Performs an economic QR decomposition with column pivoting.

* Inputs:
a: An m x n matrix to decompose.  
k: The rank of the output approximation.

* Output:
A tuple (Q, R, P) containing:
- Q: An orthogonal matrix of size m x k.
- R: An upper triangular matrix of size k x n.
- P: A vector indicating the permutation of columns applied during pivoting.

This algorithm uses Householder reflections to compute the decomposition up to the specified rank k,
which is faster and more efficient for certain applications while preserving stability.
*/
pub fn economic_qrcp(
    a: &DMatrix<f64>,
    k: usize
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let (m, n) = a.shape();
    assert!(k <= m.min(n), "k must be <= min(m,n)");
    assert!(k > 0, "k must be positive");
    let mut q = DMatrix::identity(m, m);
    let mut r = a.clone();
    let mut p: Vec<usize> = (0..n).collect();
    
    let mut col_norms: Vec<f64> = (0..n)
        .map(|j| r.column(j).norm())
        .collect();
    
    for k_step in 0..k {
        let mut max_norm = col_norms[k_step];
        let mut max_idx = k_step;
        
        for j in (k_step + 1)..n {
            if col_norms[j] > max_norm {
                max_norm = col_norms[j];
                max_idx = j;
            }
        }
        
        // Pivot
        if max_idx != k_step {
            r.swap_columns(k_step, max_idx);
            p.swap(k_step, max_idx);
            col_norms.swap(k_step, max_idx);
        }
        
        // Compute Householder
        let mut x = DVector::zeros(m - k_step);
        for i in k_step..m {
            x[i - k_step] = r[(i, k_step)];
        }
        
        let norm_x = x.norm();
        if !(norm_x == 0.0) {
            let mut v = x;
            v[0] += if v[0] >= 0.0 { norm_x } else { -norm_x };
            v /= v.norm();
            
            // Reflect R
            for j in k_step..n {
                let dot_product = v.dot(&r.view((k_step, j), (m - k_step, 1)));
                for i in k_step..m {
                    r[(i, j)] -= 2.0 * v[i - k_step] * dot_product;
                }
            }
            
            // Update Q
            let h = DMatrix::identity(m, m) - 
                2.0 * 
                DMatrix::from_fn(m, m, |i, j| {
                    if i >= k_step && j >= k_step {
                        v[i - k_step] * v[j - k_step]
                    } else {
                        0.0
                    }
                });
            q = q * h;
            
            for j in (k_step + 1)..n {
                col_norms[j] = r.view((k_step + 1, j), (m - k_step - 1, 1)).norm();
            }
        }
    }
    
    let q_eco = q.columns(0, k).into_owned();
    let r_eco = r.rows(0, k).into_owned();
    
    (q_eco, r_eco, p)
}

#[cfg(test)]
mod tests
{
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use super::{qrcp, economic_qrcp, lupp};
    use crate::sketch::{sketching_operator, DistributionType};
    use rand::Rng;
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
    fn test_qrcp(){
        let n = rand::thread_rng().gen_range(10..30);
        let m = rand::thread_rng().gen_range(n..500);
        let data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        
        let (q_cp, r_cp, p) = qrcp(&data);
        let p_cp = permutation_vector_to_transpose_matrix(&p);
        let (q, r) = data.clone().qr().unpack();
        
        let reconstruct = &q*&r;
        let reconstructed = &q_cp*&r_cp*&p_cp;
        
        // Normal columns
        let cols = q_cp.ncols();
        for j in 0..cols {
            assert_relative_eq!(q_cp.column(j).norm(), 1.0, epsilon = 1e-6);
        }

        // Orthogonal columns
        for i in 0..cols {
            for j in (i+1)..cols {
                assert_relative_eq!(q_cp.column(i).dot(&q_cp.column(j)), 0.0, epsilon = 1e-6);
            }
        }
        
        assert!(check_upper_triangular(&r_cp, 1e-4));
        assert!(check_approx_equal(&reconstructed, &reconstruct, 1e-4));
        assert!(check_approx_equal(&reconstructed, &data, 1e-4));
    }
    
    #[test]
    fn test_lupp(){
        let n = 500;
        let m = 500;
        let data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        
        let (l_rp, u_rp, p) = lupp(&data).unwrap();
        let p_rp = permutation_vector_to_transpose_matrix(&p).transpose();
        
        let (p, l, u) = data.clone().lu().unpack();
        let reconstruct = p_rp*&l_rp*&u_rp;
        let mut reconstructed = &l*&u;
        p.inv_permute_rows(&mut reconstructed);
        
        assert!(check_upper_triangular(&u_rp, 1e-4));
        assert!(check_upper_triangular(&l_rp.transpose(), 1e-4));
        assert!(check_approx_equal(&data, &reconstruct, 1e-4));
        assert!(check_approx_equal(&reconstructed, &reconstruct, 1e-4));
    }
    

    #[test]
    fn test_qrcp_economical(){
        let n = rand::thread_rng().gen_range(10..30);
        let m = rand::thread_rng().gen_range(n..500);
        let k = rand::thread_rng().gen_range(n/2..n);
        let data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        
        let (q_cp, r_cp, _) = economic_qrcp(&data, k);
        // let p_cp = permutation_vector_to_transpose_matrix(&p);
        
        let qtq = &q_cp.transpose()*&q_cp;
        let indices:Vec<usize> = (0..k).collect();
        assert!(check_upper_triangular(&r_cp.select_rows(&indices), 1e-4));
        assert!(check_approx_equal(&qtq,  &DMatrix::identity(q_cp.ncols(), q_cp.ncols()), 1e-4));
    }

    #[test]
    #[should_panic]
    fn test_qrcp_economical_bad_input(){
        let n = rand::thread_rng().gen_range(10..30);
        let m = rand::thread_rng().gen_range(n..500);
        let data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        
        economic_qrcp(&data, 0);
    }
}