use rand_core::le;

#[inline]
fn mul32(a: u32, b:u32) -> (u32, u32) {
    let prod = (a as u64).wrapping_mul(b as u64);
    ((prod >> 32) as u32, prod as u32)
}

#[inline]
fn mul64(a: u64, b:u64) -> (u64, u64) {
    let prod = (a as u128).wrapping_mul(b as u128);
    ((prod >> 64) as u64, prod as u64)
}


// multipliers and Weyl constants
const PHILOX_M2X64_0: u64 = 0xD2B74407B1CE6E93;
const PHILOX_M4X64_0: u64 = 0xD2E7470EE14C6C93;
const PHILOX_M4X64_1: u64 = 0xCA5A826395121157;
const PHILOX_W64_0: u64 = 0x9E3779B97F4A7C15;   // golden ratio
const PHILOX_W64_1: u64 = 0xBB67AE8584CAA73B;   // sqrt(3)-1

const PHILOX_M2X32_0:u32 = 0xd256d193;
const PHILOX_M4X32_0:u32 = 0xD2511F53;
const PHILOX_M4X32_1:u32 = 0xCD9E8D57;
const PHILOX_W32_0:u32 = 0x9E3779B9;
const PHILOX_W32_1:u32 = 0xBB67AE85;

pub type Array1x32 = [u32; 1];
pub type Array2x32 = [u32; 2];
pub type Array4x32 = [u32; 4];
pub type Array1x64 = [u64; 1];
pub type Array2x64 = [u64; 2];
pub type Array4x64 = [u64; 4];

#[derive(Clone)]
pub struct Philox2x32 {
    ctr: Array2x32,
    key: Array1x32,
}

impl Philox2x32 {
    pub fn next(&mut self) -> Array2x32 {
        let results = philox_2x32(self.ctr, self.key);
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] == 0 {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
        }
        results
    }
    pub fn from_seed(seed: [u8; 4]) -> Self {
        let mut key = [0u32; 1];
        le::read_u32_into(&seed, &mut key);
        Self { ctr: [0,0], key: key }
    }
}

#[derive(Clone)]
pub struct Philox2x64 {
    ctr: Array2x64,
    key: Array1x64,
}

impl Philox2x64 {
    pub fn next(&mut self) -> Array2x64 {
        let results = philox_2x64(self.ctr, self.key);
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] == 0 {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
        }
        results
    }
    pub fn from_seed(seed: [u8; 8]) -> Self {
        let mut key = [0u64; 1];
        le::read_u64_into(&seed, &mut key);
        Self { ctr: [0,0], key: key }
    }
}

#[derive(Clone)]
pub struct Philox4x32 {
    ctr: Array4x32,
    key: Array2x32,
}

impl Philox4x32 {
    pub fn next(&mut self) -> Array4x32 {
        let results = philox_4x32(self.ctr, self.key);
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] == 0 {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
            if self.ctr[1] == 0 {
                self.ctr[2] = self.ctr[2].wrapping_add(1);
                if self.ctr[2] == 0 {
                    self.ctr[3] = self.ctr[3].wrapping_add(1);
                }
            }
        }
        results
    }
    pub fn from_seed(seed: [u8; 8]) -> Self {
        let mut key = [0u32; 2];
        le::read_u32_into(&seed, &mut key);
        Self { ctr: [0,0,0,0], key: key }
    }
}

#[derive(Clone)]
pub struct Philox4x64 {
    ctr: Array4x64,
    key: Array2x64,
}

impl Philox4x64 {
    #[inline]
    pub fn next(&mut self) -> Array4x64 {
        let results = philox_4x64(self.ctr, self.key);
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] == 0 {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
            if self.ctr[1] == 0 {
                self.ctr[2] = self.ctr[2].wrapping_add(1);
                if self.ctr[2] == 0 {
                    self.ctr[3] = self.ctr[3].wrapping_add(1);
                }
            }
        }
        results
    }
    pub fn from_seed(seed: [u8; 16]) -> Self {
        let mut key = [0u64; 2];
        le::read_u64_into(&seed, &mut key);
        Self { ctr: [0,0,0,0], key: key }
    }
}

#[inline]
fn philox_2x32round(ctr: Array2x32, key: Array1x32) -> Array2x32 {
    let (hi, lo) = mul32(PHILOX_M2X32_0, ctr[0]);
    [hi ^ key[0]^ctr[1], lo]
}

#[inline]
fn philox_2x64round(ctr: Array2x64, key: Array1x64) -> Array2x64 {
    let (hi, lo) = mul64(PHILOX_M2X64_0, ctr[0]);
    [hi ^ key[0]^ctr[1], lo]
}

#[inline]
fn philox_4x32round(ctr: Array4x32, key: Array2x32) -> Array4x32 {
    let (hi0, lo0) = mul32(PHILOX_M4X32_0, ctr[0]);
    let (hi1, lo1) = mul32(PHILOX_M4X32_1, ctr[2]);
    [hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0]
}

#[inline]
fn philox_4x64round(ctr: Array4x64, key: Array2x64) -> Array4x64 {
    let (hi0, lo0) = mul64(PHILOX_M4X64_0, ctr[0]);
    let (hi1, lo1) = mul64(PHILOX_M4X64_1, ctr[2]);
    [hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0]
}

#[inline]
fn philox_1x32key(key: Array1x32) -> Array1x32 {
    [key[0].wrapping_add(PHILOX_W32_0)]
}

#[inline]
fn philox_1x64key(key: Array1x64) -> Array1x64 {
    [key[0].wrapping_add(PHILOX_W64_0)]
}

#[inline]
fn philox_2x32key(key: Array2x32) -> Array2x32 {
    [key[0].wrapping_add(PHILOX_W32_0), key[1].wrapping_add(PHILOX_W32_1)]
}

#[inline]
fn philox_2x64key(key: Array2x64) -> Array2x64 {
    [key[0].wrapping_add(PHILOX_W64_0), key[1].wrapping_add(PHILOX_W64_1)]
}

pub fn philox_2x32(ctr: Array2x32, key: Array1x32) -> Array2x32 {
                                   let ctr = philox_2x32round(ctr, key);    // 0
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 1
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 2
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 3
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 4
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 5
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 6
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 7
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 8
    let key = philox_1x32key(key); let ctr = philox_2x32round(ctr, key);    // 9
    ctr
}

pub fn philox_2x64(ctr: Array2x64, key: Array1x64) -> Array2x64 {
                                   let ctr = philox_2x64round(ctr, key);    // 0
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 1
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 2
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 3
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 4
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 5
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 6
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 7
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 8
    let key = philox_1x64key(key); let ctr = philox_2x64round(ctr, key);    // 9
    ctr
}

pub fn philox_4x32(ctr: Array4x32, key: Array2x32) -> Array4x32 {
                                   let ctr = philox_4x32round(ctr, key);    // 0
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 1
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 2
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 3
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 4
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 5
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 6
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 7
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 8
    let key = philox_2x32key(key); let ctr = philox_4x32round(ctr, key);    // 9
    ctr
}

pub fn philox_4x64(ctr: Array4x64, key: Array2x64) -> Array4x64 {
                                   let ctr = philox_4x64round(ctr, key);    // 0
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 1
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 2
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 3
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 4
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 5
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 6
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 7
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 8
    let key = philox_2x64key(key); let ctr = philox_4x64round(ctr, key);    // 9
    ctr
}


#[cfg(test)]
mod tests {
    const TEST_VEC_2X32: [u32; 20] = [
        0xa6c50e2f, 0x1588d3cf,
        0x69fa231c, 0x42a3e92d,
        0x8a9a54bc, 0x63ab381,
        0x64153a09, 0x2368eb47,
        0x1547b128, 0x5fdf6b1d,
        0x5eac56b9, 0x12601ce7,
        0xa3356439, 0x35af6c44,
        0x5388c668, 0x48435fc3,
        0xe6bc1a68, 0xc68a441c,
        0x5c1cb970, 0xc8f3d547,
    ];

    const TEST_VEC_2X64: [u64; 20] = [
        0x34675e1ecb3b1b21, 0xee5260e7a5c77077,
        0xadb9b4be0434137d, 0x93d08d9f78601509,
        0x918659c175f3632e, 0xc7cca79fda8a2af6,
        0xa52f003f96814ae6, 0x551760aa1b7bca9a,
        0x8b5225b0e1250a74, 0xe2e3a62578a061e5,
        0x66ffbbc69accd943, 0xa55d0934fbd351d6,
        0xbb6a97c67a5369c6, 0x23aaefe4dd4e8c73,
        0xa8bb64c61cf1e41e, 0xfe70f49925e4af13,
        0xfbb1dc990229435b, 0xdd17cb26850f9476,
        0x76a57ff13ece4477, 0xe9892f7aec6ffe79,
    ];

    const TEST_VEC_4X32: [u32; 40] = [
        0xcc7d356a, 0x5e7dedd7, 0x76798bc3, 0x6c05818c,
        0x4d7d84fc, 0x44ea4626, 0x26680a11, 0xc5c86681,
        0x5344ffa0, 0xa0300aea, 0x650c4611, 0xf274539d,
        0x99b25360, 0x9316678 , 0xd791ce76, 0xb12c3349,
        0xa65ceb5 , 0x9514aef7, 0xe5528b10, 0x7e6416b4,
        0x36f3b5f , 0xa78d244b, 0x192afe87, 0xfa93201,
        0x45d59db0, 0xd93533dd, 0x150a6435, 0x88f8a2e8,
        0x1882b35 , 0xe365ff23, 0x4e06c6cf, 0x4a2d3133,
        0x1ea732e6, 0x8835f7fd, 0x20219e72, 0xfde01b3a,
        0xc1c5424f, 0x1591eacd, 0x90e83125, 0x471a8bf4,
    ];

    const TEST_VEC_4X64: [u64; 40] = [
        0xfe494842771cb1b6, 0xea586dadd6341852, 0x18181e59afbb3b26, 0xff1cfc8ccd1a1bd,
        0x97179b431f365937, 0x7c15d6d2c839cafe, 0xff74c7d8fd2bec61, 0xfb9ed887767aebad,
        0xbd4e2e1d8e407bb7, 0x2da69ea34c749707, 0x31ed86c14cd4c8a7, 0xf9cdf29143c06e0d,
        0x55c5db5167f20f6 , 0x3dfa59ab31635463, 0xb9f55e0288013619, 0x729b593ab7d430c,
        0xd1c9404cc704577c, 0x3f3db5ca19acc116, 0x2d7d8724aed9b0d0, 0xb0a97c0cf2c7c8d6,
        0xa8097e7a6cb4fd63, 0xae2bfe5447a5c04d, 0xd46f4348c420ba2c, 0x85ca4cca64484952,
        0x9243eef3329c9271, 0x5f07d976cb864ae7, 0xb5f09fcdf1ed8303, 0x8419ffc8dab5ad2d,
        0x60ee58fb209c1334, 0x18595db3fc3e8cbe, 0xa27575f033fc6767, 0x67e3290de74129e5,
        0xb7994caf92d01753, 0xbc0f2d82c1b49cee, 0xa4b1bb17efb25c26, 0xbe12d4946ed88657,
        0xd7f8f58ffdad3359, 0x3f5dc9336cb23e80, 0x88870c74f15b1d34, 0x9afed694d45dc329,
    ];

    const SEED1: u32 = 0x11111111;
    const SEED2: u32 = 0x22222222;

    use super::{Array1x32, Array2x32, Array4x32};
    use super::{Array1x64, Array2x64, Array4x64};
    use super::{philox_2x32, philox_4x32, philox_2x64, philox_4x64};

    #[test]
    fn exact_values_philox_2x32() {
        let mut ctr: Array2x32 = [0,0];
        let key: Array1x32 = [SEED1];
        for i in 0..10 {
            ctr[0] = i;
            let x = philox_2x32(ctr, key);
            let i0 = (2*i+0) as usize;
            let i1 = (2*i+1) as usize;
            assert_eq!(x[0], TEST_VEC_2X32[i0]);
            assert_eq!(x[1], TEST_VEC_2X32[i1]);
        }
    }

    #[test]
    fn exact_values_philox_2x64() {
        let mut ctr: Array2x64 = [0,0];
        let key: Array1x64 = [SEED1 as u64];
        for i in 0..10 {
            ctr[0] = i;
            let x = philox_2x64(ctr, key);
            let i0 = (2*i+0) as usize;
            let i1 = (2*i+1) as usize;
            assert_eq!(x[0], TEST_VEC_2X64[i0]);
            assert_eq!(x[1], TEST_VEC_2X64[i1]);
        }
    }

    #[test]
    fn exact_values_philox_4x32() {
        let mut ctr: Array4x32 = [0,0,0,0];
        let key: Array2x32 = [SEED1, SEED2];
        for i in 0..10 {
            ctr[0] = i;
            let x = philox_4x32(ctr, key);
            let i0 = (4*i+0) as usize;
            let i1 = (4*i+1) as usize;
            let i2 = (4*i+2) as usize;
            let i3 = (4*i+3) as usize;
            assert_eq!(x[0], TEST_VEC_4X32[i0]);
            assert_eq!(x[1], TEST_VEC_4X32[i1]);
            assert_eq!(x[2], TEST_VEC_4X32[i2]);
            assert_eq!(x[3], TEST_VEC_4X32[i3]);
        }
    }

    #[test]
    fn exact_values_philox_4x64() {
        let mut ctr: Array4x64 = [0,0,0,0];
        let key: Array2x64 = [SEED1 as u64, SEED2 as u64];
        for i in 0..10 {
            ctr[0] = i;
            let x = philox_4x64(ctr, key);
            let i0 = (4*i+0) as usize;
            let i1 = (4*i+1) as usize;
            let i2 = (4*i+2) as usize;
            let i3 = (4*i+3) as usize;
            assert_eq!(x[0], TEST_VEC_4X64[i0]);
            assert_eq!(x[1], TEST_VEC_4X64[i1]);
            assert_eq!(x[2], TEST_VEC_4X64[i2]);
            assert_eq!(x[3], TEST_VEC_4X64[i3]);
        }
    }
}
