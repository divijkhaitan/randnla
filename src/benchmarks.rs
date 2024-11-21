use nalgebra::DMatrix;
use rand::Rng;
use std::time::Instant;
use criterion::{black_box, Criterion, criterion_group, criterion_main};
use plotters::prelude::*;

use crate::sketch_and_precondition;
use crate::sketch_and_solve;
use crate::sketch;
use crate::solvers;
use crate::cg;
use crate::errors;
use crate::pivot_decompositions;
use crate::id;
use crate::cqrrpt;
use crate::lora_drivers::{rand_svd, rand_evd1, rand_evd2};
use crate::lora_helpers;
use crate::reg_quad;
use crate::cg_tests;
use crate::test_assist::{generate_random_matrix, generate_random_hermitian_matrix, generate_random_psd_matrix, check_approx_equal};

// Need to benchmark mainly two things: speed and accuracy and showcase the tradeoffs. Show speed graphs with increase in matrix dimensions and accuracy graphs with increase in matrix dimensions. Also graph with the k parameter also cause that's the rank k approximation

// can plot the time and the accuracy on the same graph with two axes




#[cfg(test)]
mod svd_benchmarks {
    use super::*;

    // Benchmark structure to hold results
    struct SVDBenchmarkResult {
        matrix_size: usize,
        rand_svd_time: std::time::Duration,
        det_svd_time: std::time::Duration,
        reconstruction_error: f64,
    }

    // Function to run individual benchmark
    fn benchmark_svd(height: usize, width: usize, k: usize) -> SVDBenchmarkResult {
        let a_matrix = generate_random_matrix(height, width);
        let epsilon = 1e-6;
        let s = 2;

        // Randomized SVD
        let tick = Instant::now();
        let (u_comp, sigma, v_comp) = rand_svd(&a_matrix, k, epsilon, s);
        let rand_svd_time = tick.elapsed();

        // Deterministic SVD
        let tick = Instant::now();
        let svd = a_matrix.svd(true, true);
        let det_svd_time = tick.elapsed();

        // Extract deterministic components
        let binding = svd.u.unwrap();
        let u_det = binding.columns(0, k).clone();

        let binding = svd.v_t.unwrap().transpose();
        let v_det = binding.columns(0, k).clone().transpose();

        let s_binding = DMatrix::from_diagonal(&svd.singular_values);
        let rows_trunc = s_binding.rows(0, k);
        let s_det = rows_trunc.columns(0,k).clone();

        // Reconstruct matrices
        let approx: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = 
            &u_comp * &sigma * v_comp.clone();

        let orig_trunc = &u_det * &s_det * &v_det;

        // Calculate reconstruction error
        let reconstruction_error = (&approx - &orig_trunc).norm();

        SVDBenchmarkResult {
            matrix_size: height * width,
            rand_svd_time,
            det_svd_time,
            reconstruction_error,
        }
    }

    // Function to run comprehensive benchmarks
    fn run_comprehensive_benchmarks() -> Vec<SVDBenchmarkResult> {
        let sizes = vec![
        (10, 5),     // Very small matrix
        (20, 10),    // Small matrix
        (30, 15),    // Small matrix
        (40, 20),    // Small matrix
        (50, 25),    // Small matrix
        (60, 30),    // Small matrix
        (70, 35),    // Small matrix
        (80, 40),    // Small matrix
        (90, 45),    // Small matrix
        (100, 50),   // Small matrix
        (150, 75),   // Small to Medium matrix
        (200, 100),  // Medium matrix
        (250, 125),  // Medium matrix
        (300, 150),  // Medium matrix
        (350, 175),  // Medium matrix
        (400, 200),  // Medium matrix
        (450, 225),  // Medium matrix
        (500, 250),  // Medium matrix
        // (1000, 500), // Large matrix
        // (2000, 1000),// Very large matrix
    ];

        sizes.iter()
            .map(|&(height, width)| {
                let k = width / 2;
                benchmark_svd(height, width, k)
            })
            .collect()
    }

    // Plot performance results
    fn plot_performance_results(results: &[SVDBenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("svd_performance.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("SVD Performance Comparison", ("Arial", 30).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f64..results.last().unwrap().matrix_size as f64 * 1.1, 
                0f64..results.iter().map(|r| r.rand_svd_time.as_secs_f64().max(r.det_svd_time.as_secs_f64())).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() * 1.1
            )?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.matrix_size as f64, r.rand_svd_time.as_secs_f64())),
            &RED
        ))?
        .label("Randomized SVD Time")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.matrix_size as f64, r.det_svd_time.as_secs_f64())),
            &BLUE
        ))?
        .label("Deterministic SVD Time")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    // Plot reconstruction error
    fn plot_reconstruction_error(results: &[SVDBenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("svd_reconstruction_error.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("SVD Reconstruction Error", ("Arial", 30).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f64..results.last().unwrap().matrix_size as f64 * 1.1, 
                0f64..results.iter().map(|r| r.reconstruction_error).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() * 1.1
            )?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.matrix_size as f64, r.reconstruction_error)),
            &GREEN
        ))?
        .label("Reconstruction Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    #[test]
    pub fn comprehensive_svd_benchmark() {
        println!("\n============ Comprehensive SVD Benchmark ============\n");
        
        // Run benchmarks
        let results = run_comprehensive_benchmarks();

        // Print detailed results
        for result in &results {
            println!("Matrix Size: {}", result.matrix_size);
            println!("Randomized SVD Time: {:?}", result.rand_svd_time);
            println!("Deterministic SVD Time: {:?}", result.det_svd_time);
            println!("Reconstruction Error: {}\n", result.reconstruction_error);
        }

        // Plot results
        plot_performance_results(&results).expect("Failed to plot performance");
        plot_reconstruction_error(&results).expect("Failed to plot reconstruction error");

        println!("\n============ Benchmarking Complete! ============\n");
    }
}



#[cfg(test)]
mod evd1_benchmarks {
    use full_palette::PURPLE;

    use super::*;

    // Benchmark structure to hold results
    struct EVDBenchmarkResult {
        matrix_size: usize,
        rand_evd_time: std::time::Duration,
        det_evd_time: std::time::Duration,
        eigenvalue_reconstruction_error: f64,
        eigenvector_reconstruction_error: f64,
    }

    // Function to run individual benchmark
    fn benchmark_evd(dims: usize, k: usize) -> EVDBenchmarkResult {
        // Generate a symmetric positive definite matrix
        println!("Dims: {}", dims);
        let A_rand = generate_random_matrix(dims, dims);
        let A_rand_psd = &A_rand * &A_rand.transpose();
        
        let epsilon = 1e-6;
        let s = 2;

        // Randomized Eigenvalue Decomposition
        let tick = Instant::now();
        let (v_rand, lambda_rand) = rand_evd1(&A_rand_psd, k, epsilon, s);
        let rand_evd_time = tick.elapsed();

        // Deterministic Eigenvalue Decomposition
        let tick = Instant::now();
        let normal_evd = A_rand_psd.symmetric_eigen();
        let det_evd_time = tick.elapsed();

        // Extract top k eigenvectors and eigenvalues from deterministic EVD
        let trunc_eigvecs = normal_evd.eigenvectors.columns(0, k);
        let trunc_eigvals = normal_evd.eigenvalues.iter().take(k).cloned().collect::<Vec<_>>();

        println!("Rand EVD: {:?}", lambda_rand);
        println!("Det EVD: {:?}", trunc_eigvals);

        // Calculate reconstruction errors
        // Eigenvector reconstruction error
        let eigenvector_reconstruction_error = (&v_rand - &trunc_eigvecs).norm();

        // Eigenvalue reconstruction error
        let eigenvalue_reconstruction_error = 
            lambda_rand.iter()
                .zip(trunc_eigvals.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();

        EVDBenchmarkResult {
            matrix_size: dims * dims,
            rand_evd_time,
            det_evd_time,
            eigenvalue_reconstruction_error,
            eigenvector_reconstruction_error,
        }
    }

    // Function to run comprehensive benchmarks
    fn run_comprehensive_benchmarks() -> Vec<EVDBenchmarkResult> {
        let sizes = vec![
        (5,3),
        (6,4),
        (7,5),
        (8,6),
        (9,7),
        (10, 5),     // Very small matrix
        (20, 10),    // Small matrix
        (30, 15),    // Small matrix
        (40, 20),    // Small matrix
        (50, 25),    // Small matrix
        (60, 30),    // Small matrix
        (70, 35),    // Small matrix
        (80, 40),    // Small matrix
        (90, 45),    // Small matrix
        (100, 50),   // Small matrix
        (150, 75),   // Small to Medium matrix
        (200, 100),  // Medium matrix
        (250, 125),  // Medium matrix
        (300, 150),  // Medium matrix
        (350, 175),  // Medium matrix
        (400, 200),  // Medium matrix
        // (450, 225),  // Medium matrix
        (500, 250),  // Medium matrix
        (1000, 500), // Large matrix
        (2000, 1000),// Very large matrix
        (4000, 2000),
    ];

        sizes.iter()
            .map(|&(dims, k)| benchmark_evd(dims, k))
            .collect()
    }

    // Plot performance results
    fn plot_performance_results(results: &[EVDBenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("evd_performance.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Eigenvalue Decomposition Performance Comparison", ("Arial", 30).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f64..results.last().unwrap().matrix_size as f64 * 1.1, 
                0f64..results.iter().map(|r| r.rand_evd_time.as_secs_f64().max(r.det_evd_time.as_secs_f64())).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() * 1.1
            )?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.matrix_size as f64, r.rand_evd_time.as_secs_f64())),
            &RED
        ))?
        .label("Randomized EVD Time")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.matrix_size as f64, r.det_evd_time.as_secs_f64())),
            &BLUE
        ))?
        .label("Deterministic EVD Time")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    // Plot reconstruction errors
    fn plot_reconstruction_errors(results: &[EVDBenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("evd_reconstruction_error.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Eigenvalue Decomposition Reconstruction Error", ("Arial", 30).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f64..results.last().unwrap().matrix_size as f64 * 1.1, 
                0f64..results.iter().map(|r| r.eigenvalue_reconstruction_error.max(r.eigenvector_reconstruction_error)).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() * 1.1
            )?;

        chart.configure_mesh().draw()?;

        // Eigenvalue reconstruction error
        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.matrix_size as f64, r.eigenvalue_reconstruction_error)),
            &GREEN
        ))?
        .label("Eigenvalue Reconstruction Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        // Eigenvector reconstruction error
        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.matrix_size as f64, r.eigenvector_reconstruction_error)),
            &PURPLE
        ))?
        .label("Eigenvector Reconstruction Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &PURPLE));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    #[test]
    pub fn comprehensive_evd1_benchmark() {
        println!("\n============ Comprehensive EVD Benchmark ============\n");
        
        // Run benchmarks
        let results = run_comprehensive_benchmarks();

        // Print detailed results
        for result in &results {
            println!("Matrix Size: {}", result.matrix_size);
            println!("Randomized EVD Time: {:?}", result.rand_evd_time);
            println!("Deterministic EVD Time: {:?}", result.det_evd_time);
            println!("Eigenvalue Reconstruction Error: {}", result.eigenvalue_reconstruction_error);
            println!("Eigenvector Reconstruction Error: {}\n", result.eigenvector_reconstruction_error);
        }

        // Plot results
        plot_performance_results(&results).expect("Failed to plot performance");
        plot_reconstruction_errors(&results).expect("Failed to plot reconstruction errors");

        println!("\n============ Benchmarking Complete! ============\n");
    }
}