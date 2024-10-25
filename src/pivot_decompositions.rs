use nalgebra::{DMatrix, DVector};

fn permutation_vector_to_matrix(perm: &[usize]) -> DMatrix<f64> {
    let n = perm.len();
    let mut perm_matrix = DMatrix::<f64>::zeros(n, n); // Initialize an n x n matrix with zeros

    for (i, &p) in perm.iter().enumerate() {
        perm_matrix[(i, p)] = 1.0;
    }

    perm_matrix
}

pub fn qrcp(
    a: &DMatrix<f64>
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let (m, n) = a.shape();
    let mut q = DMatrix::identity(m, m);
    let mut r = a.clone();
    let mut p: Vec<usize> = (0..n).collect();
    
    // Compute column norms for pivoting
    let mut col_norms: Vec<f64> = (0..n)
        .map(|j| r.column(j).norm())
        .collect();
    
    for k in 0..n.min(m) {
        // Find column with maximum norm
        let mut max_norm = col_norms[k];
        let mut max_idx = k;
        
        for j in (k + 1)..n {
            if col_norms[j] > max_norm {
                max_norm = col_norms[j];
                max_idx = j;
            }
        }
        
        // Swap columns if necessary
        if max_idx != k {
            r.swap_columns(k, max_idx);
            p.swap(k, max_idx);
            col_norms.swap(k, max_idx);
        }
        
        // Compute Householder reflection
        let mut x = DVector::zeros(m - k);
        for i in k..m {
            x[i - k] = r[(i, k)];
        }
        
        let norm_x = x.norm();
        if !(norm_x == 0.0) {
            let mut v = x;
            v[0] += if v[0] >= 0.0 { norm_x } else { -norm_x };
            v /= v.norm();
            
            // Apply Householder reflection to R
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
            
            // Update column norms
            for j in (k + 1)..n {
                col_norms[j] = r.view((k + 1, j), (m - k - 1, 1)).norm();
            }
        }
    }
    
    (q, r, p)
}

#[cfg(test)]
mod tests
{
    use nalgebra::DMatrix;
    use super::qrcp;
    use crate::{pivot_decompositions::permutation_vector_to_matrix, sketch::{sketching_operator, DistributionType}};
    use rand::Rng;
    fn check_approx_equal(a: &DMatrix<f64>, b: &DMatrix<f64>, tolerance: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if (a[(i, j)] - b[(i, j)]).abs() > tolerance {
                    println!("{}, {}", i, j);
                    return false;
                }
            }
        }
        
        true
    }
    
    fn check_upper_triangular(a: &DMatrix<f64>, tolerance: f64) -> bool {
        
        for i in 0..a.nrows() {
            for j in 0..i {
                if (a[(i, j)]).abs() > tolerance {
                    println!("({}, {}), {}", i, j, a[(i, j)]);
                    return false;
                }
            }
        }
        true
    }
    #[test]
    fn test_qrcp()
    {
        let n = rand::thread_rng().gen_range(10..30);
        let m = rand::thread_rng().gen_range(n..500);
        let data = sketching_operator(DistributionType::Gaussian, m, n).unwrap();
        
        let (q_cp, r_cp, p) = qrcp(&data);
        let p_cp = permutation_vector_to_matrix(&p);
        
        let (q, r) = data.qr().unpack();
        
        let reconstruct = &q*&r;
        let reconstructed = q_cp*r_cp*p_cp;
        let qtq = &q.transpose()*&q;
        assert!(check_upper_triangular(&r, 1e-4));
        assert!(check_approx_equal(&qtq,  &DMatrix::identity(q.ncols(), q.ncols()), 1e-4));
        assert!(check_approx_equal(&reconstruct, &reconstructed, 1e-4));
    }
}