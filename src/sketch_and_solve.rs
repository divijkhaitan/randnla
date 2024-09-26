use nalgebra::DMatrix;
use crate::sketch::{sketching_operator, DistributionType};
use crate::solvers::solve_upper_triangular_system;

pub fn sketched_least_squares_qr(a:&DMatrix<f64>, b:&DMatrix<f64>) -> DMatrix<f64>
{
    let rows = a.nrows();
    let s: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = sketching_operator(DistributionType::Gaussian, rows/2, rows);
    let a_sk = &s*a;
    let mut b_sk = &s*b;
    let (q , r) = a_sk.qr().unpack();
    b_sk = q.transpose()*b_sk;
    let x = solve_upper_triangular_system(&r, &b_sk);
    x
}


#[cfg(test)]
mod tests
{
    use num_traits::Pow;
    use rand::Rng;
    use rand_distr::{Uniform, Normal, Distribution};
    use nalgebra::DMatrix;
    use super::sketched_least_squares_qr;
    use crate::{sketch::{sketching_operator, DistributionType}, solvers::solve_upper_triangular_system};
    #[test]
    fn test_least_squares()
    {
        // This code is to generate a random hypothesis, and add generate noisy data from that hypothesis
        let mut rng = rand::thread_rng();
        let n = rand::thread_rng().gen_range(10..30);
        let m = rand::thread_rng().gen_range(n..500);
        let epsilon = 0.01;
        // let mut noise: [i32;n] = [0;n];
        let normal = Normal::new(0.0, epsilon).unwrap();
        let uniform = Uniform::new(-100.0, 100.0);
        let hypothesis = DMatrix::from_fn(n, 1, |_i, _j| uniform.sample(&mut rng));
        let mut data = sketching_operator(DistributionType::Gaussian, m, n);
        let y = &data*&hypothesis;
        for i in 0..m {
            let noise_vector = DMatrix::from_fn(n, 1, |_, _| normal.sample(&mut rng));
            for j in 0..n {
                data[(i, j)] += noise_vector[(j, 0)];
            }
        }
        // compute using sketched qr
        let x = sketched_least_squares_qr(&data, &y);
        
        // compute using plain qr
        let (q, r) = data.qr().unpack();
        let b_transformed = q.transpose()*y;
        let actual_solution = solve_upper_triangular_system(&r, &b_transformed);
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        let mut norm3 = 0.0;
        let mut norm4 = 0.0;
        let mut norm5 = 0.0;
        let mut norm6 = 0.0;
        for i in 0..x.ncols(){
            norm1+= (hypothesis[(i, 0)] - actual_solution[(i, 0)]).pow(2.0);
            norm2+= (actual_solution[(i, 0)] - x[(i, 0)]).pow(2.0);
            norm3+= (x[(i, 0)] - hypothesis[(i, 0)]).pow(2.0);
            norm4+= (hypothesis[(i, 0)]).pow(2.0);
            norm5+= (actual_solution[(i, 0)]).pow(2.0);
            norm6+= (x[(i, 0)]).pow(2.0);
        }
        println!("{}, {}, {}, {}, {}, {}", norm1, norm2, norm3, norm4, norm5, norm6);
        // assert!((norm1 < 0.2));
        // assert!((norm2 < 0.2));
        // assert!((norm3 < 0.2));
        // println!("Hypothesis: \n{}", hypothesis);
        // println!("Unsketched Solution: \n{}", actual_solution);

    }
}