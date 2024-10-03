use nalgebra::{DMatrix, DVector};

pub fn conjugate_grad(a: &DMatrix<f64>, b: &DVector<f64>, x: Option<DVector<f64>>) -> DVector<f64> {
    let n = b.len();
    let mut x = x.unwrap_or_else(|| DVector::from_element(n, 1.0));
    let mut r = a * &x - b;
    let mut p = -&r;
    let mut r_k_norm = r.dot(&r);

    for i in 0..(2 * n) {
        let ap = a * &p;
        let alpha = r_k_norm / p.dot(&ap);
        x += alpha * &p;
        r += alpha * ap;
        let r_kplus1_norm = r.dot(&r);
        
        if r_kplus1_norm < 1e-10 {
            println!("Converged after {} iterations", i);
            break;
        }
        
        let beta = r_kplus1_norm / r_k_norm;
        r_k_norm = r_kplus1_norm;
        p = beta * p - &r;
    }

    x
}

pub fn verify_solution(a: &DMatrix<f64>, b: &DVector<f64>, x: &DVector<f64>) -> f64 {
    (a * x - b).norm()
}