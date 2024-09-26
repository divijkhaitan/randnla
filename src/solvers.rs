use nalgebra::{DMatrix, Scalar, RealField};
use num_traits::{Zero, One};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

pub fn solve_upper_triangular_system<T>(u: &DMatrix<T>, y: &DMatrix<T>) -> DMatrix<T>
where
    T: Scalar + RealField + Zero + One + AddAssign + SubAssign + MulAssign + DivAssign + Copy,
{
    let n = u.nrows();
    let mut x = DMatrix::zeros(n, 1);
    
    for i in (0..n).rev() {
        let mut sum = T::zero();
        for j in (i+1)..n {
            sum += u[(i, j)] * x[(j, 0)];
        }
        
        x[(i, 0)] = (y[(i, 0)] - sum) / u[(i, i)];
    }
    x
}