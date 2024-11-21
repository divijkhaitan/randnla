/*!
 * A package for randomised numerical linear algebra written in Rust.
*/
#![allow(dead_code)]
// #![allow(unused_imports)]
#![allow(non_snake_case)]

pub mod sketch_and_precondition;
pub mod sketch_and_solve;
pub mod sketch;
pub mod solvers;
pub mod cg;
pub mod errors;
pub mod pivot_decompositions;
pub mod id;
pub mod cqrrpt;
pub mod lora_drivers;
pub mod lora_helpers;
pub mod test_assist;
// pub mod benchmarks;