# randblas

A package for randomized numerical linear algebra written in Rust by Divij Khaitan and Saptarishi Dhanuka based on [Randomized Numerical Linear Algebra : A Perspective on the Field With an Eye to Software] monograph (https://arxiv.org/abs/2302.11474) and the corresponding C++ codebase [RandLAPACK](https://github.com/BallisticLA/RandLAPACK)

<!-- ## Examples

Examples for all the main library features are present in the [main.rs](./src/main.rs) file -->

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

### Linear Algebra Operations
- `src/cg.rs`: Conjugate gradient method for solving linear systems
  - `conjugate_grad()`: Implementation of conjugate gradient solver
  - `cgls()`
  - `verify_solution()`: Verification routine for CG solutions

### Matrix Decompositions  
- `src/id.rs`: Interpolative Decomposition (ID) implementations
  - `cur()`: CUR matrix decomposition
  - `cur_randomised()`: Randomized CUR decomposition
  - `two_sided_id()`: Two-sided interpolative decomposition
  - `two_sided_id_randomised()`: Randomized two-sided ID

### Low-Rank Approximations
- `src/lora_drivers.rs`: Main driver functions
  - `rand_svd()`: Randomized SVD implementation
  - `rand_evd1()`: First randomized eigenvalue decomposition variant
  - `rand_evd2()`: Second randomized eigenvalue decomposition variant

### Solvers


### Testing & Benchmarks
- `src/benchmarks.rs`: Performance benchmarking
  - SVD benchmarks 
  - Eigenvalue decomposition benchmarks
- `src/test_assist.rs`: Testing utilities
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
The full documentation can be found in the [randblas](./doc/randblas/) folder in `html` files directory or by running ```cargo doc --open```

