use core::fmt;
use rand_core::{RngCore, SeedableRng, Error as RandCoreError};
use rand_core::block::{BlockRngCore, BlockRng, BlockRng64};

use super::philox::{Philox2x32,  Philox2x64,  Philox4x32,  Philox4x64};
use super::threefry::ThreeFry2x64;

//
extern crate rand;
//

macro_rules! impl_rng {
    ($t: ty, $n:expr, $i: ty, $b: expr, $block: ident, $rng: path) => {

        impl BlockRngCore for $t {
            type Item = $i;
            type Results = [$i; $n];

            fn generate(&mut self, results: &mut Self::Results) {
                *results = self.next();
            }
        }

        impl SeedableRng for $t {
            type Seed = [u8; $b];

            fn from_seed(seed: Self::Seed) -> Self {
                <$t>::from_seed(seed)
            }
        }

        impl fmt::Debug for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{} {{}}", stringify!($t))
            }
        }

        impl SeedableRng for $rng {
            type Seed = <$t as SeedableRng>::Seed;

            fn from_seed(seed: Self::Seed) -> Self {
                $rng($block::<$t>::from_seed(seed))
            }

            fn from_rng<R: RngCore>(rng: R) -> Result<Self, RandCoreError> {
                $block::<$t>::from_rng(rng).map($rng)
            }
        }

        impl From<$t> for $rng {
            fn from(core: $t) -> Self {
                $rng($block::new(core))
            }
        }

        impl RngCore for $rng {
            #[inline]
            fn next_u32(&mut self) -> u32 {
                self.0.next_u32()
            }
        
            #[inline]
            fn next_u64(&mut self) -> u64 {
                self.0.next_u64()
            }
        
            #[inline]
            fn fill_bytes(&mut self, dest: &mut [u8]) {
                self.0.fill_bytes(dest)
            }
        
            #[inline]
            fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RandCoreError> {
                self.0.try_fill_bytes(dest)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Philox2x32Rng(BlockRng<Philox2x32>);
#[derive(Clone, Debug)]
pub struct Philox2x64Rng(BlockRng64<Philox2x64>);
#[derive(Clone, Debug)]
pub struct Philox4x32Rng(BlockRng<Philox4x32>);
#[derive(Clone, Debug)]
pub struct Philox4x64Rng(BlockRng64<Philox4x64>);
#[derive(Clone, Debug)]
pub struct ThreeFry2x64Rng(BlockRng64<ThreeFry2x64>);

impl_rng!(Philox2x32, 2, u32, 4, BlockRng,   Philox2x32Rng);
impl_rng!(Philox2x64, 2, u64, 8, BlockRng64, Philox2x64Rng);
impl_rng!(Philox4x32, 4, u32, 8, BlockRng,   Philox4x32Rng);
impl_rng!(Philox4x64, 4, u64,16, BlockRng64, Philox4x64Rng);

impl_rng!(ThreeFry2x64, 2, u64,16, BlockRng64, ThreeFry2x64Rng);

//
// impl rand::RngCore for ThreeFry2x64Rng {
//     #[inline]
//     fn next_u32(&mut self) -> u32 {
//         self.0.next_u32()
//     }

//     #[inline]
//     fn next_u64(&mut self) -> u64 {
//         self.0.next_u64()
//     }

//     #[inline]
//     fn fill_bytes(&mut self, dest: &mut [u8]) {
//         self.0.fill_bytes(dest)
//     }

//     #[inline]
//     fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
//         self.0.try_fill_bytes(dest)
//     }
// }


// impl rand::Rng for ThreeFry2x64Rng {}
//

#[cfg(test)]
mod tests {

    use super::ThreeFry2x64Rng;
    use rand_core::{SeedableRng, RngCore};

    #[test]
    fn fill_bytes() {
        // more of an example really

        let mut buf = [0u8; 64];
        let mut rng = ThreeFry2x64Rng::seed_from_u64(42);
        rng.fill_bytes(&mut buf);
    }
}

