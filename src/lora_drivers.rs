use nalgebra::{DMatrix};
use crate::lora_helpers;







// randomized SVD
pub fn rand_SVD(A:&DMatrix<f64>, k:usize, epsilon: f64, s: usize) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)
{
    println!("Running RSVD");
    /*
    Q, B = QBDecomposer(A, k+s, epsilon)
    r = min(k, Q.ncols())
    U, S, V = svd(B)
    U = U[:, :r]
    V = V[: , : r]
    S = S[: r , : r]
    U = QU
    return U, S, V
     */
    
    // TODO: check the positive and negative switching of the values
    let (Q,B) = lora_helpers::QB1(&A, k+s, epsilon);
    let r = std::cmp::min(k, Q.ncols());
    let mysvd_rand = B.clone().svd(true, true);
    let U_binding = mysvd_rand.u.unwrap();
    let U = U_binding.columns(0, r).clone();

    let V_binding = mysvd_rand.v_t.unwrap().transpose();
    let V = V_binding.columns(0, r).clone();

    let S_binding = DMatrix::from_diagonal(&mysvd_rand.singular_values);
    let S = S_binding.rows(0, r).clone();
    let U_final = Q*U;
    return (U_final, S.into(), V.into());


}