use nalgebra::DMatrix;
use crate::sketch::{sketching_operator, MatrixAttribute, DistributionType};
use crate::lora_helpers;
use crate::pivot_decompositions::economic_qrcp;

pub fn cur(
    a: &DMatrix<f64>,
    k: usize
) -> (Vec<usize>, DMatrix<f64>, Vec<usize>) {
    let (m, n) = a.shape();
    
    if m >= n {
        let (x, j) = osid_qrcp(a, k, MatrixAttribute::Column);
        
        let a_cols = a.select_columns(&j);
        
        let (_, _, i) = economic_qrcp(&a_cols.transpose(), k);
        
        let i = i[0..k].to_vec();
        
        // Compute U = X(A[I,:])^T
        let a_rows = a.select_rows(&i);
        let u = x * a_rows.pseudo_inverse(0.0).unwrap();
        
        (j[0..k].to_vec(), u, i)
    } else {
        let a_t = a.transpose();
        let (z, i) = osid_qrcp(&a_t, k, MatrixAttribute::Column);
        
        let a_rows = a.select_rows(&i);
        
        // Perform QR decomposition
        let (_, _, j) = economic_qrcp(&a_rows, k);
        
        // Take first k indices
        let j = j[0..k].to_vec();
        
        // Compute U = (A[:,J])^T Z
        let a_cols = a.select_columns(&j);
        let a_pseudoinv = a_cols.pseudo_inverse(0.0).unwrap();
        let u = a_pseudoinv * z.transpose();
        
        (j, u, i[0..k].to_vec())
    }
}

pub fn two_sided_id_randomised(
    a: &DMatrix<f64>,
    k: usize,
) -> (DMatrix<f64>, Vec<usize>, Vec<usize>, DMatrix<f64>) {
    let (x, j) = osid_randomised(a, k, MatrixAttribute::Column);
    let (z, i) = osid_randomised(&a.select_columns(&j), k, MatrixAttribute::Row);
    (z, i, j, x)
}

pub fn two_sided_id(
    a: &DMatrix<f64>,
    k: usize,
) -> (DMatrix<f64>, Vec<usize>, Vec<usize>, DMatrix<f64>) {
    let (x, j) = osid_qrcp(a, k, MatrixAttribute::Column);
    let (z, i) = osid_qrcp(&a.select_columns(&j), k, MatrixAttribute::Row);
    (z, i, j, x)
}

pub fn cur_randomised(
    a: &DMatrix<f64>,
    k: usize
) -> (Vec<usize>, DMatrix<f64>, Vec<usize>) {
    let (m, n) = a.shape();
    
    if m >= n {
        let (x, j) = osid_randomised(a, k, MatrixAttribute::Column);
        
        let a_cols = a.select_columns(&j);
        
        let (_, _, i) = economic_qrcp(&a_cols.transpose(), k);
        
        let i = i[0..k].to_vec();
        
        // Compute U = X(A[I,:])^T
        let a_rows = a.select_rows(&i);
        let u = x * a_rows.pseudo_inverse(0.0).unwrap();
        
        (j[0..k].to_vec(), u, i)
    } else {
        let a_t = a.transpose();
        let (z, i) = osid_randomised(&a_t, k, MatrixAttribute::Column);
        
        let a_rows = a.select_rows(&i);
        
        // Perform QR decomposition
        let (_, _, j) = economic_qrcp(&a_rows, k);
        
        // Take first k indices
        let j = j[0..k].to_vec();
        
        // Compute U = (A[:,J])^T Z
        let a_cols = a.select_columns(&j);
        let a_pseudoinv = a_cols.pseudo_inverse(0.0).unwrap();
        let u = a_pseudoinv * z.transpose();
        
        (j, u, i[0..k].to_vec())
    }
}

pub fn osid_randomised(
    a: &DMatrix<f64>,
    k: usize,
    attr: MatrixAttribute,
) -> (DMatrix<f64>, Vec<usize>) {
    let (m, n) = a.shape();
    assert!(k > 0, "k must be positive)");
    assert!(k <= m.min(n), "k must be less than min(m,n)");
    
    match attr {
        MatrixAttribute::Row => {
            // Generate sketch operator S
            let s_matrix = sketching_operator(DistributionType::Gaussian, a.ncols(), k).unwrap();
            // let s_matrix = lora_helpers::tsog1(&a, k, 0, 2, 1);
            
            // Compute Y = AS
            let y = a * s_matrix.transpose();
            
            // Get row ID of Y
            osid_qrcp(&y, k, MatrixAttribute::Row)
        },
        MatrixAttribute::Column => {
            // Generate sketch operator S for A^T
            let s_matrix = sketching_operator(DistributionType::Gaussian, k, a.nrows()).unwrap();
            // let s_matrix = lora_helpers::tsog1(&a.transpose(), k, 0, 2, 1);
            // Compute Y = SA
            let y = s_matrix * a;
            
            // Get column ID of Y
            osid_qrcp(&y, k, MatrixAttribute::Column)
        }
    }
}

pub fn osid_qrcp(
    y: &DMatrix<f64>,
    k: usize,
    attr: MatrixAttribute,
) -> (DMatrix<f64>, Vec<usize>) {
    let (l, w) = y.shape();
    assert!(k <= l.min(w), "k must be <= min(l,w)");

    match attr {
        MatrixAttribute::Column => {
            let (_, r, j) = economic_qrcp(y, k);
            
            let r1 = r.view((0,0), (k,k));
            let r2 = r.view((0,k), (k, w-k));
            
            // Solve triangular system: T = R1^{-1} R2
            let t = r1.solve_upper_triangular(&r2).unwrap();
            
            // Construct linking matrix
            let mut x = DMatrix::zeros(k, w);
            let i_k = DMatrix::identity(k, k);
            
            // Set X[:, J[0:k]] = I_k
            for (idx, &j_idx) in j.iter().take(k).enumerate() {
                for i in 0..k {
                    x[(i, j_idx)] = i_k[(i, idx)];
                }
            }
            
            // Set remaining columns using T
            for (col_idx, &j_idx) in j.iter().skip(k).enumerate() {
                for i in 0..k {
                    x[(i, j_idx)] = t[(i, col_idx)];
                }
            }
            
            (x, j[0..k].to_vec())
        },
        MatrixAttribute::Row => {
            // Row ID - transpose and use column ID
            let y_t = y.transpose();
            let (x, i) = osid_qrcp(&y_t, k, MatrixAttribute::Column);
            (x.transpose(), i)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{id::{two_sided_id, two_sided_id_randomised}, sketch::MatrixAttribute};
    use nalgebra::DMatrix;
    use super::{osid_qrcp, cur, cur_randomised};
    use rand::{Rng, thread_rng};
    use rand_distr::{Distribution, StandardNormal};
    use std::time::Instant;
    fn rank_k_matrix(m: usize, n: usize, k: usize) -> DMatrix<f64> {
        assert!(k <= m.min(n), "k must be <= min(m,n)");
        
        let mut rng = thread_rng();
        let normal = StandardNormal;
    
        // Preallocate output matrix
        let mut result = DMatrix::zeros(m, n);
        
        // Generate and multiply k rank-1 updates
        for _ in 0..k {
            // Generate random vectors
            let u: Vec<f64> = (0..m).map(|_| normal.sample(&mut rng)).collect();
            let v: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            
            // Add rank-1 update: u * v^T
            for i in 0..m {
                for j in 0..n {
                    result[(i, j)] += u[i] * v[j];
                }
            }
        }
        
        result
    }
    
    #[test]
    fn test_one_sided_id()
    {
        let m = rand::thread_rng().gen_range(100..500);
        let n = rand::thread_rng().gen_range(100..500);
        let k = rand::thread_rng().gen_range(m.min(n)/2..m.min(n));
        // let mut rng = thread_rng();
        // let uniform = Uniform::new(-100.0, 100.0);
        
        // let matrix = DMatrix::from_fn(m, n, |_i, _j| uniform.sample(&mut rng));
        let matrix = rank_k_matrix(m, n, k);
        let start_deterministic = Instant::now();
        let (x, indices) = osid_qrcp(&matrix, k, MatrixAttribute::Column);
        let y_subset = matrix.select_columns(&indices);
        let y_approx = y_subset * x;
        let duration_deterministic = start_deterministic.elapsed();

        let start_randomised = Instant::now();
        let (x_randomised, indices) = osid_qrcp(&matrix, k, MatrixAttribute::Column);
        let y_subset = matrix.select_columns(&indices);
        let y_random_approx = y_subset * x_randomised;
        let duration_randomised = start_randomised.elapsed();
        
        println!("Difference Deterministic: {}", (&y_approx-&matrix).norm()/(matrix.norm()));
        println!("Difference Randomised: {}", (&y_random_approx-&matrix).norm()/(matrix.norm()));
        println!("Difference Between Approximations: {}", (&y_approx-&y_random_approx).norm());
        println!("Deterministic Time: {:.2?}, Randomised Time:{:.2?}", duration_deterministic, duration_randomised);
        println!("{}", k);
    }
    #[test]
    fn test_two_sided_id()
    {
        let m = rand::thread_rng().gen_range(100..500);
        let n = rand::thread_rng().gen_range(100..500);
        let k = rand::thread_rng().gen_range(m.min(n)/2..m.min(n));
        let matrix = rank_k_matrix(m, n, k);
        
        // Deterministic ID
        let start_deterministic = Instant::now();
        let (z, i, j, x) = two_sided_id(&matrix, k);
        let y_subset = matrix.select_rows(&i).select_columns(&j);
        let y_approx = &z * &y_subset * &x;
        let duration_deterministic = start_deterministic.elapsed();
        
        // Randomised ID
        let start_randomised = Instant::now();
        let (z_random, i_random, j_random, x_random) = two_sided_id_randomised(&matrix, k);
        let y_subset_random = matrix.select_rows(&i_random).select_columns(&j_random);
        let y_approx_random = &z_random * &y_subset_random * &x_random;
        let duration_randomised = start_randomised.elapsed();
        
        println!("Difference (Deterministic, Randomised): {}, {}", (&y_approx-&matrix).norm()/(matrix.norm()), (&y_approx_random-&matrix).norm()/(matrix.norm()));
        println!("Difference Between Approximations: {}", (&y_approx-&y_approx_random).norm());
        println!("Time (Deterministic, Randomised): {:.2?} {:.2?}", duration_deterministic, duration_randomised);
        println!("{}", k);
    }
    #[test]
    fn test_cur()
    {
        let m = rand::thread_rng().gen_range(100..500);
        let n = rand::thread_rng().gen_range(100..500);
        let k = rand::thread_rng().gen_range(m.min(n)/2..m.min(n));
        let matrix = rank_k_matrix(m, n, k);
        
        // Deterministic CUR
        let start_deterministic = Instant::now();
        let (c, u, r) = cur(&matrix, k);
        let y_column_subset = matrix.select_columns(&c);
        let y_row_subset = matrix.select_rows(&r);
        let y_approx = y_column_subset * &u * y_row_subset;
        let duration_deterministic = start_deterministic.elapsed();
        
        // Randomised CUR
        let start_randomised = Instant::now();
        let (c_random, u_random, r_random) = cur_randomised(&matrix, k);
        let y_column_subset_random = matrix.select_columns(&c_random);
        let y_row_subset_random = matrix.select_rows(&r_random);
        let y_approx_random = y_column_subset_random * &u_random * y_row_subset_random;
        let duration_randomised = start_randomised.elapsed();
        
        println!("Difference (Deterministic, Randomised): {}, {}", (&y_approx-&matrix).norm()/(matrix.norm()), (&y_approx_random-&matrix).norm()/(matrix.norm()));
        println!("Difference Between Approximations: {}", (&y_approx-&y_approx_random).norm());
        println!("Time (Deterministic, Randomised): {:.2?} {:.2?}", duration_deterministic, duration_randomised);
        println!("{}", k);
    }
    
}
