
extern crate rand_123;
extern crate rand_core;

use rand_123::rng::ThreeFry2x64Rng;
use rand_core::{SeedableRng, RngCore};

fn main() {
    let mut rng = ThreeFry2x64Rng::seed_from_u64(0);
    loop {
        println!("{}", rng.next_u64());
    }
}

