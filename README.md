# randblas

A package for randomized numerical linear algebra written in Rust by Divij Khaitan and Saptarishi Dhanuka based on [Randomized Numerical Linear Algebra : A Perspective on the Field With an Eye to Software](https://arxiv.org/abs/2302.11474) monograph  and the corresponding C++ codebase [RandLAPACK](https://github.com/BallisticLA/RandLAPACK)

## Examples

Examples for all the main library features are present in the correspondint test cases for each of the `.rs` files

<!-- ## Repository Structure
src/ 
    benchmarks.rs # Benchmark implementations for SVD and EVD 

    cg.rs # Conjugate gradient method implementations 

    cqrrpt.rs # Column QR with pivoting implementations

    errors.rs # Error type definitions 

    id.rs # Interpolative decomposition implementations 

    lib.rs # Main library file 

    lora_drivers.rs # Low-rank approximation driver functions 

    lora_helpers.rs # Helper functions for low-rank approximations 

    test_assist.rs # Testing utility functions -->



## Core Components

### Sketching-related
- [`sketch`](./src/sketch.rs): Sketching operators
- [`sketch_and_precondition`](/src/sketch_and_precondition.rs): For solving overdetermined least squares problems
- [`sketch_and_solve`](./src/sketch_and_solve.rs): For solving overdetermined least squares problems but rougher approximations that `sketch_and_precondition`


### Solvers
- [`cg.rs`](./src/cg.rs): Conjugate gradient methods for solving linear systems
  - `conjugate_grad()`: Implementation of conjugate gradient solver for Ax = b
  - `cgls()`: Similar to conjugate_grad but more stable in finite precision arithmetic.
  - `verify_solution()`: Verification routine for CG solutions
- [`solvers.rs`](./src/solvers.rs): Various solvers for linear systems
    - `solve_upper_triangular_system()`: For upper triangular A
    - `solve_diagonal_system()`: For diagonal A
    - `lsqr()`: More stable than `conjugate_grad()` and `cgls()` for ill-conditioned problems

### Matrix Decompositions  
- [`id.rs`](./src/id.rs): Interpolative Decomposition (ID) implementations
  - `cur()`: CUR matrix decomposition
  - `cur_randomised()`: Randomized CUR decomposition
  - `two_sided_id()`: Two-sided interpolative decomposition
  - `two_sided_id_randomised()`: Randomized two-sided ID
- [`pivot_decompositions`](./src/pivot_decompositions.rs): Decompositions with column pivoting

### Low-Rank Approximations
- [`lora_drivers.rs`](./src/lora_drivers.rs): Main driver functions
  - `rand_svd()`: Randomized SVD implementation
  - `rand_evd1()`: First randomized eigendecomposition variant
  - `rand_evd2()`: Second randomized eigendecomposition variant
- [`lora_helpers.rs`](./src/lora_helpers.rs): Helpers for drivers

### Testing & Benchmarks
- [`benchmarks.rs`](./src/benchmarksrs): Performance benchmarking `(__INCOMPLETE RIGHT NOW__ since not needed by deadline)`
  - SVD benchmarks 
  - Eigendecomposition benchmarks
- [`main.rs`](./src/main.rs): Contains some more performance related code (times out after 5 min)
- [`test_assist.rs`](./src/lora_drivers.rs): Testing utilities
  - Matrix generation functions
  - Numerical verification routines

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/divijkhaitan/randnla.git
```
2. View documentation
```bash
cargo doc --open
```
3. Build the project
```bash
cargo build
```
4. Run tests
```bash
cargo test
```

## Documentation
The full documentation can be found in the [randblas](./doc/randblas/) folder in `html` files residing in folders for each file, or by running ```cargo doc --open``` which will open the documentation in the browser where you can navigate more easily

