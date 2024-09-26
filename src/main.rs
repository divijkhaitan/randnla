use sketch::haar_sample;

mod sketch;
mod sketch_and_solve;
mod solvers;
fn main() {
    let n = 6;
    let row_haar_matrix = haar_sample(n, n, sketch::MatrixAttribute::Row);
    println!("Row Orthogonal:\n{}", row_haar_matrix);
    
    let col_haar_matrix = haar_sample(n, n, sketch::MatrixAttribute::Column);
    println!("Column Orthogonal:\n{}", col_haar_matrix);
    
    let gaussian_matrix = sketch::sketching_operator(sketch::DistributionType::Gaussian, n, n);
    println!("Gaussian Matrix:\n{}", gaussian_matrix);
    
    let uniform_matrix = sketch::sketching_operator(sketch::DistributionType::Uniform, n, n);
    println!("Uniform Matrix:\n{}", uniform_matrix);
    
    let rademacher_matrix = sketch::sketching_operator(sketch::DistributionType::Rademacher, n, n);
    println!("Rademacher Matrix:\n{}", rademacher_matrix);
}