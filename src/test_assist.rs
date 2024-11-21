use nalgebra::{DMatrix, DVector};
use rand_distr::{Distribution, Normal, Uniform, StandardNormal};
use rand::{thread_rng, Rng};
use crate::sketch::{sketching_operator, DistributionType};

/// Generates a matrix of rank k of a given size using rank 1 updates
pub fn rank_k_matrix(m: usize, n: usize, k: usize) -> DMatrix<f64> {
    assert!(k <= m.min(n), "k must be <= min(l,w)");
    
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

/// Generates a tall matrix with a large condition number of a given size
pub fn generate_tall_ill_conditioned_matrix(m: usize, n: usize, condition_number: f64) -> DMatrix<f64> {
    assert!(m > n, "m must be greater than n for a tall matrix");
    
    let mut rng = rand::thread_rng();
    let mut a = sketching_operator(DistributionType::Gaussian, m, n).unwrap();

    // Modified Gram-Schmidt
    for i in 0..n {
        let mut v = a.column(i).clone_owned();
        for j in 0..i {
            let proj = a.column(j).dot(&v);
            v -= proj * a.column(j);
        }
        v /= v.norm();
        a.set_column(i, &v);
    }

    // Scale columns to worsen conditioning
    let singular_values = DVector::from_fn(n, |i, _| {
        if i == 0 {
            condition_number
        } else if i == n - 1 {
            1.0
        } else {
            rng.gen_range(1.0..condition_number)
        }
    });

    for i in 0..n {
        let mut col = a.column_mut(i);
        col *= singular_values[i];
    }

    a
}

/// Generates a least squares problem of a given size
pub fn generate_least_squares_problem(m:usize , n:usize, ill_conditioning:bool)  -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
    let mut rng = rand::thread_rng();
    let epsilon = 0.0001;
    let normal = Normal::new(0.0, epsilon).unwrap();
    let uniform = Uniform::new(-100.0, 100.0);
    let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng));
    let data = {
        if ill_conditioning{
            generate_tall_ill_conditioned_matrix(m, n, 1e6)
        }
        else
        {
            sketching_operator(DistributionType::Gaussian, m, n).unwrap()
        }
    };
    let mut y = &data*&hypothesis;
    let noise_vector = DMatrix::from_fn(m, 1, |_, _| normal.sample(&mut rng));
    for i in 0..m {
        y[(i, 0)] += noise_vector[(i, 0)];
    }
    (data, hypothesis, y)
}


/// Returns the permutation matrix transposed from QRCP
pub fn permutation_vector_to_transpose_matrix(perm: &[usize]) -> DMatrix<f64> {
    let n = perm.len();
    let mut perm_matrix = DMatrix::<f64>::zeros(n, n); // Initialize an n x n matrix with zeros

    for (i, &p) in perm.iter().enumerate() {
        perm_matrix[(i, p)] = 1.0;
    }

    perm_matrix
}

/// Checks if the lower triangle of a matrix is zero within some tolerance
pub fn check_upper_triangular(a: &DMatrix<f64>, tolerance: f64) -> bool {
    
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


/// Generates a random matrix of size (rows, cols) with normally distributed elems
pub fn generate_random_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| normal.sample(&mut rng)).collect();
    DMatrix::from_vec(rows, cols, data)
}

/// Generates a random Hermitian matrix of size (n, n) with normally distributed elems
pub fn generate_random_hermitian_matrix(n: usize) -> DMatrix<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..n * n).map(|_| normal.sample(&mut rng)).collect();
    let mut matrix = DMatrix::from_vec(n, n, data);
    let matrix_t = matrix.transpose();
    matrix = 0.5 * (matrix + matrix_t); // symmetric
    matrix
}

/// Generates a random psd matrix of size (n, n) with normally distributed elems
pub fn generate_random_psd_matrix(n: usize) -> DMatrix<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..n * n).map(|_| normal.sample(&mut rng)).collect();
    let mut matrix = DMatrix::from_vec(n, n, data);
    let matrix_t = matrix.transpose();
    matrix = matrix * matrix_t; 
    matrix
}

/// Checks if two matrices are equal within some tolerance
pub fn check_approx_equal(a: &DMatrix<f64>, b: &DMatrix<f64>, tolerance: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if (a[(i, j)] - b[(i, j)]).abs() > tolerance {
                // println!("{}, {}, {}, {}", i, j, a[(i, j)], b[(i, j)]);
                return false;
            }
        }
    }
    
    true
}