//! bf16 ↔ f32, as the CPU engine does them.

pub type Bf16 = u16;

/// Exact widening.
#[inline(always)]
pub fn bf16_to_f32(h: Bf16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Round to nearest even (`f32_to_bf16` of herbert-jit-bf16, bit for bit).
#[inline(always)]
pub fn f32_to_bf16(v: f32) -> Bf16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let round_bit = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let mut result = bits >> 16;
    if round_bit == 1 && (sticky != 0 || lsb == 1) {
        result += 1;
    }
    result as u16
}

/// The bytes of a slice of plain numbers, little-endian as the GPU reads them.
pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: u16/u32/f32 have no padding and every byte pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
