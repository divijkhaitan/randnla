mod sample;

fn main() {
    let n = 6;
    let gaussian_matrix = sample::sketching_operator(sample::DistributionType::Gaussian, n, n);
    println!("Gaussian Matrix:\n{}", gaussian_matrix);
    
    let uniform_matrix = sample::sketching_operator(sample::DistributionType::Uniform, n, n);
    println!("Uniform Matrix:\n{}", uniform_matrix);
    
    let rademacher_matrix = sample::sketching_operator(sample::DistributionType::Rademacher, n, n);
    println!("Rademacher Matrix:\n{}", rademacher_matrix);
}