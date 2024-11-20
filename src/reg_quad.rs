#![allow(dead_code)]
#![allow(warnings)]
#![allow(unused_imports)]

use nalgebra::{DMatrix, DVector};
use crate::sketch::{sketching_operator, DistributionType};
use crate::cg;
use crate::lora_drivers;
use std::error::Error;



// pub fn minimize_reg_quad(A: &DMatrix<f64>, b: &DMatrix<f64>, lambda: f64) -> DMatrix<f64> {
//     // presume access to low rank approximation which we can get through rand evd svd or whatever
//     // then define a preconditioner
//     // approximating G with the Nystrom approximation and then solving saddle point or smth problem with the sketch and solve or sketch and precondition methods already available to us
// }


fn nystrom_approx(g: &DMatrix<f64>, r: usize, mu: f64) -> (DMatrix<f64>, DMatrix<f64>) {

    let (eigs_v, lambda) = lora_drivers::rand_evd2(g, r, 0).unwrap();



    return (DMatrix::zeros(1, 1), DMatrix::zeros(1, 1));

}