//! The PTX emitter: typed registers and one method per instruction, writing
//! the text of one kernel (one module, one `.entry`) at a time.
//!
//! Every kernel of the engine is a function `xxx(shape…) -> Emitted` built
//! with [`Kernel`]: the dimensions known at load time are written as
//! immediates, the few values that change from call to call (the rows of a
//! prefill block, the position) are kernel parameters. No CUDA C, no NVRTC:
//! the text below is the module, and the driver's ptxas compiles it
//! (`cuModuleLoadDataEx`). `ptxas -arch=sm_89` validates it offline.
//!
//! What the layer gives:
//! - five register classes, as nvcc names them: [`P`] (`.pred`), [`H`]
//!   (`.b16`), [`R`] (`.b32`, every 32-bit integer and packed f16 pair),
//!   [`F`] (`.f32`), [`D`] (`.b64`, addresses). A bit-size instruction may
//!   take any register of its size (PTX's type rules);
//! - registers are mutable, as PTX's are: each method returns a fresh one,
//!   the `*_to` / in-place methods write a given one (loop-carried values,
//!   accumulators); ptxas allocates;
//! - operands [`U`] (a register or a `u32` immediate) and [`Fv`] (a register
//!   or an `f32` immediate, written in hex: no decimal rounding);
//! - structured control flow (`if_`, `for_`) over labels and `bra`;
//! - the Ampere/Ada tensor path: `cp.async` (with zero fill), `ldmatrix`,
//!   `mma.sync.m16n8k16` f16 → f32;
//! - f16 ↔ f32 (herbert-cuda-f16): widening by `cvt.f32.f16` (exact),
//!   narrowing by `cvt.rn` (round to nearest even; beyond 65 504, ±inf).

use std::fmt::{self, Display, Write as _};

/// The PTX ISA and target every module declares (toolkit 12.8's ptxas reads
/// 8.7; the 595 driver more).
pub const VERSION: &str = "8.5";
pub const TARGET: &str = "sm_89";

/// A predicate register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct P(u32);
/// A 16-bit register (one f16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H(u32);
/// A 32-bit register: integers, bit patterns, f16 pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R(u32);
/// An f32 register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F(u32);
/// A 64-bit register: addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D(u32);

impl Display for P {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%p{}", self.0)
    }
}
impl Display for H {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%h{}", self.0)
    }
}
impl Display for R {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%r{}", self.0)
    }
}
impl Display for F {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%f{}", self.0)
    }
}
impl Display for D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%rd{}", self.0)
    }
}

/// A 32-bit integer operand: a register or an immediate.
#[derive(Clone, Copy, Debug)]
pub enum U {
    R(R),
    I(u32),
}
impl From<R> for U {
    fn from(r: R) -> U {
        U::R(r)
    }
}
impl From<u32> for U {
    fn from(i: u32) -> U {
        U::I(i)
    }
}
impl Display for U {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            U::R(r) => r.fmt(f),
            U::I(i) => write!(f, "{i}"),
        }
    }
}

/// An f32 operand: a register or an immediate (hex, exact).
#[derive(Clone, Copy, Debug)]
pub enum Fv {
    F(F),
    I(f32),
}
impl From<F> for Fv {
    fn from(r: F) -> Fv {
        Fv::F(r)
    }
}
impl From<f32> for Fv {
    fn from(i: f32) -> Fv {
        Fv::I(i)
    }
}
impl Display for Fv {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Fv::F(r) => r.fmt(f),
            Fv::I(v) => write!(f, "0f{:08X}", v.to_bits()),
        }
    }
}

/// A branch target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(u32);
impl Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

/// Integer comparisons (unsigned).
#[derive(Clone, Copy, Debug)]
pub enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}
impl Cmp {
    fn name(self) -> &'static str {
        match self {
            Cmp::Lt => "lt",
            Cmp::Le => "le",
            Cmp::Gt => "gt",
            Cmp::Ge => "ge",
            Cmp::Eq => "eq",
            Cmp::Ne => "ne",
        }
    }
}

/// The cache behaviour of a global load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ld {
    /// `ld.global`: cached in L1 and L2.
    Ca,
    /// `ld.global.nc`: the read-only path (data the kernel never writes).
    Nc,
    /// `ld.global.nc.L1::no_allocate`: streamed weights, L1 left to the rest.
    NcNoL1,
    /// `ld.global.cg`: L2 only.
    Cg,
    /// `ld.global.cs`: streaming, evict first.
    Cs,
}
impl Ld {
    fn op(self) -> &'static str {
        match self {
            Ld::Ca => "ld.global",
            Ld::Nc => "ld.global.nc",
            Ld::NcNoL1 => "ld.global.nc.L1::no_allocate",
            Ld::Cg => "ld.global.cg",
            Ld::Cs => "ld.global.cs",
        }
    }
}

/// `[a]`, `[a+off]` or `[a-off]`.
fn mem(a: impl Display, off: i32) -> String {
    match off {
        0 => format!("[{a}]"),
        o if o > 0 => format!("[{a}+{o}]"),
        o => format!("[{a}-{}]", -(o as i64)),
    }
}

/// A kernel parameter's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// A device pointer (`.u64`), converted to a global address on entry.
    Ptr,
    U32,
    F32,
}

/// One kernel being written.
pub struct Kernel {
    name: String,
    threads: u32,
    params: Vec<Param>,
    /// Static shared arrays: (symbol, bytes).
    shared: Vec<(String, u32)>,
    dynamic_shared: bool,
    body: String,
    /// The predicate of the next instruction ([`Kernel::when`]).
    guard: Option<String>,
    next: [u32; 5],
    labels: u32,
}

const CP: usize = 0;
const CH: usize = 1;
const CR: usize = 2;
const CF: usize = 3;
const CD: usize = 4;

impl Kernel {
    /// A kernel `name` launched with exactly `threads` threads per block
    /// (`.reqntid`, one dimension).
    pub fn new(name: impl Into<String>, threads: u32) -> Kernel {
        Kernel { name: name.into(), threads, params: Vec::new(), shared: Vec::new(), dynamic_shared: false, body: String::new(), guard: None, next: [0; 5], labels: 0 }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn threads(&self) -> u32 {
        self.threads
    }

    // ── registers and raw text ───────────────────────────────────────────────

    pub fn p(&mut self) -> P {
        self.next[CP] += 1;
        P(self.next[CP] - 1)
    }
    pub fn h(&mut self) -> H {
        self.next[CH] += 1;
        H(self.next[CH] - 1)
    }
    pub fn r(&mut self) -> R {
        self.next[CR] += 1;
        R(self.next[CR] - 1)
    }
    pub fn f(&mut self) -> F {
        self.next[CF] += 1;
        F(self.next[CF] - 1)
    }
    pub fn d(&mut self) -> D {
        self.next[CD] += 1;
        D(self.next[CD] - 1)
    }

    /// One instruction, as written (no `;`).
    pub fn ins(&mut self, text: impl AsRef<str>) {
        match self.guard.take() {
            Some(g) => {
                let _ = writeln!(self.body, "\t{g} {};", text.as_ref());
            }
            None => {
                let _ = writeln!(self.body, "\t{};", text.as_ref());
            }
        }
    }

    /// The next instruction runs only where `p` holds (`@p`).
    pub fn when(&mut self, p: P) {
        self.guard = Some(format!("@{p}"));
    }
    /// The next instruction runs only where `p` does not hold (`@!p`).
    pub fn unless(&mut self, p: P) {
        self.guard = Some(format!("@!{p}"));
    }

    /// A comment line in the module, to find one's way in the dump.
    pub fn note(&mut self, text: impl AsRef<str>) {
        let _ = writeln!(self.body, "\t// {}", text.as_ref());
    }

    // ── parameters and special registers ─────────────────────────────────────

    fn param_name(&self, i: usize) -> String {
        format!("{}_p{}", self.name, i)
    }

    /// The next parameter, a device pointer, as a global address.
    pub fn param_ptr(&mut self) -> D {
        let i = self.params.len();
        self.params.push(Param::Ptr);
        let (a, g) = (self.d(), self.d());
        let pn = self.param_name(i);
        self.ins(format!("ld.param.u64 {a}, [{pn}]"));
        self.ins(format!("cvta.to.global.u64 {g}, {a}"));
        g
    }

    pub fn param_u32(&mut self) -> R {
        let i = self.params.len();
        self.params.push(Param::U32);
        let r = self.r();
        let pn = self.param_name(i);
        self.ins(format!("ld.param.u32 {r}, [{pn}]"));
        r
    }

    pub fn param_f32(&mut self) -> F {
        let i = self.params.len();
        self.params.push(Param::F32);
        let r = self.f();
        let pn = self.param_name(i);
        self.ins(format!("ld.param.f32 {r}, [{pn}]"));
        r
    }

    /// A special register (`tid.x`, `ctaid.y`, `laneid`…).
    pub fn sreg(&mut self, name: &str) -> R {
        let r = self.r();
        self.ins(format!("mov.u32 {r}, %{name}"));
        r
    }
    pub fn tid(&mut self) -> R {
        self.sreg("tid.x")
    }
    pub fn ctaid_x(&mut self) -> R {
        self.sreg("ctaid.x")
    }
    pub fn ctaid_y(&mut self) -> R {
        self.sreg("ctaid.y")
    }
    pub fn nctaid_x(&mut self) -> R {
        self.sreg("nctaid.x")
    }

    // ── moves and integers (u32) ─────────────────────────────────────────────

    pub fn mov(&mut self, a: impl Into<U>) -> R {
        let r = self.r();
        self.mov_to(r, a);
        r
    }
    pub fn mov_to(&mut self, dst: R, a: impl Into<U>) {
        let a = a.into();
        self.ins(format!("mov.u32 {dst}, {a}"));
    }

    fn bin(&mut self, op: &str, a: impl Into<U>, b: impl Into<U>) -> R {
        let (a, b) = (a.into(), b.into());
        let r = self.r();
        self.ins(format!("{op} {r}, {a}, {b}"));
        r
    }
    pub fn add(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("add.u32", a, b)
    }
    pub fn add_to(&mut self, dst: R, a: impl Into<U>, b: impl Into<U>) {
        let (a, b) = (a.into(), b.into());
        self.ins(format!("add.u32 {dst}, {a}, {b}"));
    }
    pub fn sub(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("sub.u32", a, b)
    }
    pub fn mul(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("mul.lo.u32", a, b)
    }
    pub fn mad(&mut self, a: impl Into<U>, b: impl Into<U>, c: impl Into<U>) -> R {
        let (a, b, c) = (a.into(), b.into(), c.into());
        let r = self.r();
        self.ins(format!("mad.lo.u32 {r}, {a}, {b}, {c}"));
        r
    }
    pub fn shl(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("shl.b32", a, b)
    }
    pub fn shr(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("shr.u32", a, b)
    }
    pub fn and(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("and.b32", a, b)
    }
    pub fn or(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("or.b32", a, b)
    }
    pub fn xor(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("xor.b32", a, b)
    }
    pub fn min(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("min.u32", a, b)
    }
    pub fn max(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("max.u32", a, b)
    }
    pub fn div(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("div.u32", a, b)
    }
    pub fn rem(&mut self, a: impl Into<U>, b: impl Into<U>) -> R {
        self.bin("rem.u32", a, b)
    }
    /// `p ? a : b`.
    pub fn selp(&mut self, a: impl Into<U>, b: impl Into<U>, p: P) -> R {
        let (a, b) = (a.into(), b.into());
        let r = self.r();
        self.ins(format!("selp.b32 {r}, {a}, {b}, {p}"));
        r
    }

    pub fn setp(&mut self, c: Cmp, a: impl Into<U>, b: impl Into<U>) -> P {
        let (a, b) = (a.into(), b.into());
        let p = self.p();
        self.ins(format!("setp.{}.u32 {p}, {a}, {b}", c.name()));
        p
    }
    pub fn pand(&mut self, a: P, b: P) -> P {
        let p = self.p();
        self.ins(format!("and.pred {p}, {a}, {b}"));
        p
    }
    pub fn pnot(&mut self, a: P) -> P {
        let p = self.p();
        self.ins(format!("not.pred {p}, {a}"));
        p
    }

    // ── addresses (64-bit) ───────────────────────────────────────────────────

    /// `base + i·scale` (bytes), `i` widened unsigned.
    pub fn index(&mut self, base: D, i: impl Into<U>, scale: u32) -> D {
        let i = i.into();
        let (w, a) = (self.d(), self.d());
        match i {
            U::I(v) => {
                let off = v as u64 * scale as u64;
                self.ins(format!("add.s64 {a}, {base}, {off}"));
                return a;
            }
            U::R(_) => self.ins(format!("mul.wide.u32 {w}, {i}, {scale}")),
        }
        self.ins(format!("add.s64 {a}, {base}, {w}"));
        a
    }

    /// `base + bytes`.
    pub fn offset(&mut self, base: D, bytes: i64) -> D {
        let a = self.d();
        self.ins(format!("add.s64 {a}, {base}, {bytes}"));
        a
    }

    // ── floats ───────────────────────────────────────────────────────────────

    pub fn fmov(&mut self, a: impl Into<Fv>) -> F {
        let r = self.f();
        self.fmov_to(r, a);
        r
    }
    pub fn fmov_to(&mut self, dst: F, a: impl Into<Fv>) {
        let a = a.into();
        self.ins(format!("mov.f32 {dst}, {a}"));
    }
    fn fbin(&mut self, op: &str, a: impl Into<Fv>, b: impl Into<Fv>) -> F {
        let (a, b) = (a.into(), b.into());
        let r = self.f();
        self.ins(format!("{op} {r}, {a}, {b}"));
        r
    }
    pub fn fadd(&mut self, a: impl Into<Fv>, b: impl Into<Fv>) -> F {
        self.fbin("add.rn.f32", a, b)
    }
    pub fn fsub(&mut self, a: impl Into<Fv>, b: impl Into<Fv>) -> F {
        self.fbin("sub.rn.f32", a, b)
    }
    pub fn fmul(&mut self, a: impl Into<Fv>, b: impl Into<Fv>) -> F {
        self.fbin("mul.rn.f32", a, b)
    }
    pub fn fmax(&mut self, a: impl Into<Fv>, b: impl Into<Fv>) -> F {
        self.fbin("max.f32", a, b)
    }
    pub fn fmin(&mut self, a: impl Into<Fv>, b: impl Into<Fv>) -> F {
        self.fbin("min.f32", a, b)
    }
    /// `a·b + c`, one rounding.
    pub fn ffma(&mut self, a: impl Into<Fv>, b: impl Into<Fv>, c: impl Into<Fv>) -> F {
        let r = self.f();
        self.ffma_to(r, a, b, c);
        r
    }
    pub fn ffma_to(&mut self, dst: F, a: impl Into<Fv>, b: impl Into<Fv>, c: impl Into<Fv>) {
        let (a, b, c) = (a.into(), b.into(), c.into());
        self.ins(format!("fma.rn.f32 {dst}, {a}, {b}, {c}"));
    }
    pub fn fadd_to(&mut self, dst: F, a: impl Into<Fv>, b: impl Into<Fv>) {
        let (a, b) = (a.into(), b.into());
        self.ins(format!("add.rn.f32 {dst}, {a}, {b}"));
    }
    /// IEEE division and square root (`.rn`), as the CPU engine computes them.
    pub fn fdiv(&mut self, a: impl Into<Fv>, b: impl Into<Fv>) -> F {
        self.fbin("div.rn.f32", a, b)
    }
    pub fn fsqrt(&mut self, a: impl Into<Fv>) -> F {
        let a = a.into();
        let r = self.f();
        self.ins(format!("sqrt.rn.f32 {r}, {a}"));
        r
    }
    pub fn fsetp(&mut self, c: Cmp, a: impl Into<Fv>, b: impl Into<Fv>) -> P {
        let (a, b) = (a.into(), b.into());
        let p = self.p();
        self.ins(format!("setp.{}.f32 {p}, {a}, {b}", c.name()));
        p
    }
    /// The bits of an f32 as a `.b32`, and back.
    pub fn fbits(&mut self, a: F) -> R {
        let r = self.r();
        self.ins(format!("mov.b32 {r}, {a}"));
        r
    }
    pub fn bits_f(&mut self, a: R) -> F {
        let r = self.f();
        self.ins(format!("mov.b32 {r}, {a}"));
        r
    }
    pub fn u32_to_f(&mut self, a: R) -> F {
        let r = self.f();
        self.ins(format!("cvt.rn.f32.u32 {r}, {a}"));
        r
    }

    // ── 16-bit elements: IEEE f16 (herbert-cuda-f16) ────────────────────────
    // The `h_*` primitives were herbert-cuda-bf16's `bf16_*`: every 16-bit
    // element of the engine goes through them, so this block sets the format.

    /// The two halves of an f16x2 pair, widened to f32 (low, high).
    pub fn f16x2_to_f32(&mut self, pair: R) -> (F, F) {
        let (l, h) = (self.h(), self.h());
        self.ins(format!("mov.b32 {{{l}, {h}}}, {pair}"));
        let (a, b) = (self.f(), self.f());
        self.ins(format!("cvt.f32.f16 {a}, {l}"));
        self.ins(format!("cvt.f32.f16 {b}, {h}"));
        (a, b)
    }
    /// The low f16 of a pair (lower address), widened (exact).
    pub fn h_lo(&mut self, pair: R) -> F {
        let (l, h) = (self.h(), self.h());
        self.ins(format!("mov.b32 {{{l}, {h}}}, {pair}"));
        let r = self.f();
        self.ins(format!("cvt.f32.f16 {r}, {l}"));
        r
    }
    /// The high f16 of a pair, widened (exact).
    pub fn h_hi(&mut self, pair: R) -> F {
        let (l, h) = (self.h(), self.h());
        self.ins(format!("mov.b32 {{{l}, {h}}}, {pair}"));
        let r = self.f();
        self.ins(format!("cvt.f32.f16 {r}, {h}"));
        r
    }
    /// One f16, widened (exact).
    pub fn h_to_f(&mut self, h: H) -> F {
        let r = self.f();
        self.ins(format!("cvt.f32.f16 {r}, {h}"));
        r
    }
    /// Round to nearest even (IEEE f16; beyond 65 504, ±inf).
    pub fn f_to_h(&mut self, a: F) -> H {
        let h = self.h();
        self.ins(format!("cvt.rn.f16.f32 {h}, {a}"));
        h
    }
    /// IEEE f16, rounded to nearest even (the f16 experiment, `gemm.f16`).
    pub fn f_to_f16(&mut self, a: F) -> H {
        let h = self.h();
        self.ins(format!("cvt.rn.f16.f32 {h}, {a}"));
        h
    }
    /// Two f32 rounded to nearest even into one f16 pair, `lo` at the lower address.
    pub fn pack_f16(&mut self, lo: F, hi: F) -> R {
        let r = self.r();
        self.ins(format!("cvt.rn.f16x2.f32 {r}, {hi}, {lo}"));
        r
    }
    /// [`Kernel::pack_h`]; the `f16` flag of herbert-cuda-bf16's f16 experiment is moot here.
    pub fn pack16(&mut self, lo: F, hi: F, f16: bool) -> R {
        if f16 { self.pack_f16(lo, hi) } else { self.pack_h(lo, hi) }
    }
    /// Two f32 rounded to nearest even into one f16 pair: `lo` at the lower address.
    pub fn pack_h(&mut self, lo: F, hi: F) -> R {
        let r = self.r();
        self.ins(format!("cvt.rn.f16x2.f32 {r}, {hi}, {lo}"));
        r
    }

    // ── global memory ────────────────────────────────────────────────────────

    pub fn ld_v4(&mut self, c: Ld, a: D, off: i32) -> [R; 4] {
        let v = [self.r(), self.r(), self.r(), self.r()];
        self.ins(format!("{}.v4.u32 {{{}, {}, {}, {}}}, {m}", c.op(), v[0], v[1], v[2], v[3], m = mem(a, off)));
        v
    }
    pub fn ld_u32(&mut self, c: Ld, a: D, off: i32) -> R {
        let r = self.r();
        self.ins(format!("{}.u32 {r}, {m}", c.op(), m = mem(a, off)));
        r
    }
    pub fn ld_f32(&mut self, c: Ld, a: D, off: i32) -> F {
        let r = self.f();
        self.ins(format!("{}.f32 {r}, {m}", c.op(), m = mem(a, off)));
        r
    }
    pub fn ld_b16(&mut self, c: Ld, a: D, off: i32) -> H {
        let r = self.h();
        self.ins(format!("{}.b16 {r}, {m}", c.op(), m = mem(a, off)));
        r
    }
    pub fn st_v4(&mut self, a: D, off: i32, v: [R; 4]) {
        self.ins(format!("st.global.v4.u32 {m}, {{{}, {}, {}, {}}}", v[0], v[1], v[2], v[3], m = mem(a, off)));
    }
    pub fn st_u32(&mut self, a: D, off: i32, v: impl Into<U>) {
        let v = v.into();
        self.ins(format!("st.global.u32 {m}, {v}", m = mem(a, off)));
    }
    pub fn st_f32(&mut self, a: D, off: i32, v: F) {
        self.ins(format!("st.global.f32 {m}, {v}", m = mem(a, off)));
    }
    pub fn st_v2_f32(&mut self, a: D, off: i32, v: [F; 2]) {
        self.ins(format!("st.global.v2.f32 {m}, {{{}, {}}}", v[0], v[1], m = mem(a, off)));
    }
    pub fn st_b16(&mut self, a: D, off: i32, v: H) {
        self.ins(format!("st.global.b16 {m}, {v}", m = mem(a, off)));
    }

    // ── shared memory ────────────────────────────────────────────────────────

    /// A static shared array of `bytes`, 16-byte aligned; its address.
    pub fn shared(&mut self, bytes: u32) -> R {
        let sym = format!("{}_s{}", self.name, self.shared.len());
        self.shared.push((sym.clone(), bytes));
        let r = self.r();
        self.ins(format!("mov.u32 {r}, {sym}"));
        r
    }
    /// The dynamic shared memory (sized at launch), 16-byte aligned; its address.
    pub fn dyn_shared(&mut self) -> R {
        self.dynamic_shared = true;
        let r = self.r();
        let sym = format!("{}_dyn", self.name);
        self.ins(format!("mov.u32 {r}, {sym}"));
        r
    }
    pub fn lds_u32(&mut self, a: impl Into<U>, off: i32) -> R {
        let a = a.into();
        let r = self.r();
        self.ins(format!("ld.shared.u32 {r}, {m}", m = mem(a, off)));
        r
    }
    pub fn lds_f32(&mut self, a: impl Into<U>, off: i32) -> F {
        let a = a.into();
        let r = self.f();
        self.ins(format!("ld.shared.f32 {r}, {m}", m = mem(a, off)));
        r
    }
    /// A shared load ptxas may not merge with an earlier one at the same
    /// address (`ld.volatile`): to keep a value from staying live across a
    /// long stretch when reloading it is cheaper than holding it.
    pub fn lds_f32_again(&mut self, a: impl Into<U>, off: i32) -> F {
        let a = a.into();
        let r = self.f();
        self.ins(format!("ld.volatile.shared.f32 {r}, {m}", m = mem(a, off)));
        r
    }
    pub fn lds_v4(&mut self, a: impl Into<U>, off: i32) -> [R; 4] {
        let a = a.into();
        let v = [self.r(), self.r(), self.r(), self.r()];
        self.ins(format!("ld.shared.v4.u32 {{{}, {}, {}, {}}}, {m}", v[0], v[1], v[2], v[3], m = mem(a, off)));
        v
    }
    pub fn sts_u32(&mut self, a: impl Into<U>, off: i32, v: impl Into<U>) {
        let (a, v) = (a.into(), v.into());
        self.ins(format!("st.shared.u32 {m}, {v}", m = mem(a, off)));
    }
    pub fn sts_f32(&mut self, a: impl Into<U>, off: i32, v: F) {
        let a = a.into();
        self.ins(format!("st.shared.f32 {m}, {v}", m = mem(a, off)));
    }
    pub fn sts_v4(&mut self, a: impl Into<U>, off: i32, v: [R; 4]) {
        let a = a.into();
        self.ins(format!("st.shared.v4.u32 {m}, {{{}, {}, {}, {}}}", v[0], v[1], v[2], v[3], m = mem(a, off)));
    }
    pub fn bar(&mut self) {
        self.ins("bar.sync 0");
    }
    /// The warp's lanes wait for each other; shared memory written before is
    /// visible to the warp after.
    pub fn bar_warp(&mut self) {
        self.ins("bar.warp.sync 0xffffffff");
    }

    // ── warps ────────────────────────────────────────────────────────────────

    /// `a` from lane `lane ^ mask`.
    pub fn shfl_xor(&mut self, a: F, mask: u32) -> F {
        let r = self.f();
        self.ins(format!("shfl.sync.bfly.b32 {r}, {a}, {mask}, 31, 0xffffffff"));
        r
    }
    pub fn shfl_xor_u(&mut self, a: R, mask: u32) -> R {
        let r = self.r();
        self.ins(format!("shfl.sync.bfly.b32 {r}, {a}, {mask}, 31, 0xffffffff"));
        r
    }
    /// The sum of `a` over the warp, in every lane (butterfly, 5 steps).
    pub fn warp_sum(&mut self, a: F) -> F {
        let mut s = a;
        for m in [16, 8, 4, 2, 1] {
            let o = self.shfl_xor(s, m);
            s = self.fadd(s, o);
        }
        s
    }
    /// The max of `a` over the warp, in every lane.
    pub fn warp_max(&mut self, a: F) -> F {
        let mut s = a;
        for m in [16, 8, 4, 2, 1] {
            let o = self.shfl_xor(s, m);
            s = self.fmax(s, o);
        }
        s
    }

    /// The sum of `a` over the block, in every thread: each warp reduces, its
    /// lane 0 writes the partial to `scratch` (shared, one f32 per warp), and
    /// every thread adds the partials in warp order (the same result in every
    /// thread, whatever the scheduling). Barriers before and after: `scratch`
    /// may be reused at once.
    pub fn block_sum(&mut self, a: F, scratch: R) -> F {
        self.block_fold(a, scratch, false)
    }
    pub fn block_max(&mut self, a: F, scratch: R) -> F {
        self.block_fold(a, scratch, true)
    }
    fn block_fold(&mut self, a: F, scratch: R, max: bool) -> F {
        let warps = self.threads.div_ceil(32);
        let s = if max { self.warp_max(a) } else { self.warp_sum(a) };
        if warps == 1 {
            return s;
        }
        let tid = self.tid();
        let lane = self.and(tid, 31u32);
        let warp = self.shr(tid, 5u32);
        let first = self.setp(Cmp::Eq, lane, 0u32);
        let at = self.mad(warp, 4u32, scratch);
        self.bar();
        self.when(first);
        self.sts_f32(at, 0, s);
        self.bar();
        let mut t = self.lds_f32(scratch, 0);
        for w in 1..warps {
            let v = self.lds_f32(scratch, (4 * w) as i32);
            t = if max { self.fmax(t, v) } else { self.fadd(t, v) };
        }
        self.bar();
        t
    }

    // ── transcendental functions ─────────────────────────────────────────────

    /// `eˣ` as `2^(x·log₂e)` (`ex2.approx`, the hardware's; denormals flushed).
    pub fn fexp(&mut self, x: impl Into<Fv>) -> F {
        let t = self.fmul(x, std::f32::consts::LOG2_E);
        let r = self.f();
        self.ins(format!("ex2.approx.ftz.f32 {r}, {t}"));
        r
    }
    /// `ln x` as `log₂(x)·ln 2` (`lg2.approx`).
    pub fn flog(&mut self, x: impl Into<Fv>) -> F {
        let x = x.into();
        let l = self.f();
        self.ins(format!("lg2.approx.ftz.f32 {l}, {x}"));
        self.fmul(l, std::f32::consts::LN_2)
    }
    /// `1 / (1 + e⁻ˣ)`, the division IEEE.
    pub fn fsigmoid(&mut self, x: F) -> F {
        let n = self.fneg(x);
        let e = self.fexp(n);
        let d = self.fadd(e, 1.0);
        self.fdiv(1.0, d)
    }
    pub fn fneg(&mut self, x: F) -> F {
        let r = self.f();
        self.ins(format!("neg.f32 {r}, {x}"));
        r
    }
    /// `1/√(x)` computed as `1 / sqrt.rn(x)` with an IEEE division.
    pub fn finv_sqrt(&mut self, x: F) -> F {
        let r = self.fsqrt(x);
        self.fdiv(1.0, r)
    }
    /// `f16(x)` widened back: what a store and a load would give.
    pub fn round_h(&mut self, x: F) -> F {
        let h = self.f_to_h(x);
        self.h_to_f(h)
    }
    /// `p ? a : b` on floats.
    pub fn fselp(&mut self, a: impl Into<Fv>, b: impl Into<Fv>, p: P) -> F {
        let (a, b) = (a.into(), b.into());
        let r = self.f();
        self.ins(format!("selp.f32 {r}, {a}, {b}, {p}"));
        r
    }

    // ── the tensor path (sm_80+) ─────────────────────────────────────────────

    /// 16 bytes global → shared, asynchronous, through L2 only. With `bytes`
    /// (0 or 16), the source size: 0 fills the destination with zeros.
    pub fn cp_async(&mut self, dst: impl Into<U>, off: i32, src: D, bytes: Option<R>) {
        let m = mem(dst.into(), off);
        match bytes {
            None => self.ins(format!("cp.async.cg.shared.global {m}, [{src}], 16")),
            Some(n) => self.ins(format!("cp.async.cg.shared.global {m}, [{src}], 16, {n}")),
        }
    }
    pub fn cp_commit(&mut self) {
        self.ins("cp.async.commit_group");
    }
    /// Wait until at most `n` committed groups are pending.
    pub fn cp_wait(&mut self, n: u32) {
        self.ins(format!("cp.async.wait_group {n}"));
    }

    /// Ampere/Ada mbarriers in CTA shared memory (PTX 7.1+).
    pub fn mbarrier_init(&mut self, addr: R, count: u32) {
        self.ins(format!("mbarrier.init.shared.b64 [{addr}], {count}"));
    }
    /// Initialization accounts for exactly one async arrival per thread.
    pub fn cp_mbarrier_arrive(&mut self, addr: R) {
        self.ins(format!("cp.async.mbarrier.arrive.noinc.shared.b64 [{addr}]"));
    }
    pub fn mbarrier_arrive(&mut self, addr: R) {
        self.ins(format!("mbarrier.arrive.release.cta.shared.b64 _, [{addr}]"));
    }
    pub fn mbarrier_wait(&mut self, addr: R, parity: R) {
        let top = self.label();
        let done = self.p();
        self.place(top);
        self.ins(format!("mbarrier.test_wait.parity.acquire.cta.shared.b64 {done}, [{addr}], {parity}"));
        self.unless(done);
        self.ins("nanosleep.u32 32");
        self.bra_unless(done, top);
    }

    /// Four 8×8 b16 matrices from shared memory: lanes 8j…8j+7 give the row
    /// addresses of matrix j; register j holds matrix j, lane t its row t/4,
    /// columns 2(t%4) and 2(t%4)+1 (`trans`: transposed).
    pub fn ldmatrix_x4(&mut self, a: impl Into<U>, off: i32, trans: bool) -> [R; 4] {
        let a = mem(a.into(), off);
        let v = [self.r(), self.r(), self.r(), self.r()];
        let t = if trans { ".trans" } else { "" };
        self.ins(format!("ldmatrix.sync.aligned.m8n8.x4{t}.shared.b16 {{{}, {}, {}, {}}}, {a}", v[0], v[1], v[2], v[3]));
        v
    }
    pub fn ldmatrix_x2(&mut self, a: impl Into<U>, off: i32, trans: bool) -> [R; 2] {
        let a = mem(a.into(), off);
        let v = [self.r(), self.r()];
        let t = if trans { ".trans" } else { "" };
        self.ins(format!("ldmatrix.sync.aligned.m8n8.x2{t}.shared.b16 {{{}, {}}}, {a}", v[0], v[1]));
        v
    }

    /// `acc += A·B`, A 16×16 f16 row-major (4 pairs), B 16×8 f16
    /// column-major (2 pairs), acc 16×8 f32. Lane t, g = t/4, q = t%4:
    /// A pairs (row g, cols 2q..), (g+8, 2q..), (g, 2q+8..), (g+8, 2q+8..);
    /// B pairs (rows 2q.., col g), (2q+8.., g); acc (g, 2q), (g, 2q+1),
    /// (g+8, 2q), (g+8, 2q+1).
    pub fn mma_h(&mut self, acc: &[F; 4], a: &[R; 4], b: &[R; 2]) {
        self.ins(format!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{c0}, {c1}, {c2}, {c3}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{c0}, {c1}, {c2}, {c3}}}",
            a[0],
            a[1],
            a[2],
            a[3],
            b[0],
            b[1],
            c0 = acc[0],
            c1 = acc[1],
            c2 = acc[2],
            c3 = acc[3]
        ));
    }

    /// [`Kernel::mma_h`] with IEEE f16 operands: same fragments, f32 acc.
    pub fn mma_f16(&mut self, acc: &[F; 4], a: &[R; 4], b: &[R; 2]) {
        self.ins(format!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{c0}, {c1}, {c2}, {c3}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{c0}, {c1}, {c2}, {c3}}}",
            a[0], a[1], a[2], a[3], b[0], b[1], c0 = acc[0], c1 = acc[1], c2 = acc[2], c3 = acc[3]
        ));
    }
    /// f16 operands and f16 accumulators: acc is two f16x2, lane t holds
    /// (g, 2q..2q+1) in acc[0] and (g+8, 2q..2q+1) in acc[1].
    pub fn mma_f16_f16(&mut self, acc: &[R; 2], a: &[R; 4], b: &[R; 2]) {
        self.ins(format!(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {{{c0}, {c1}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{c0}, {c1}}}",
            a[0], a[1], a[2], a[3], b[0], b[1], c0 = acc[0], c1 = acc[1]
        ));
    }

    /// `wmma` (the PTX counterpart of Vulkan's cooperative matrices): the
    /// 16×16 f16 A tile, row-major, `ldm` elements per row, from shared
    /// memory; the fragment's layout is the driver's.
    pub fn wmma_load_a_h(&mut self, a: R, ldm: u32) -> [R; 4] {
        let v = [self.r(), self.r(), self.r(), self.r()];
        self.ins(format!("wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {{{}, {}, {}, {}}}, [{a}], {ldm}", v[0], v[1], v[2], v[3]));
        v
    }
    /// The 16×16 f16 B tile, column-major (rows of W), `ldm` elements apart.
    pub fn wmma_load_b_h(&mut self, a: R, ldm: u32) -> [R; 4] {
        let v = [self.r(), self.r(), self.r(), self.r()];
        self.ins(format!("wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 {{{}, {}, {}, {}}}, [{a}], {ldm}", v[0], v[1], v[2], v[3]));
        v
    }
    /// `acc += A·B`, 16×16×16, f32 accumulators in the driver's layout.
    pub fn wmma_mma_h(&mut self, acc: &[F; 8], a: &[R; 4], b: &[R; 4]) {
        let c = acc.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", ");
        self.ins(format!(
            "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {{{c}}}, {{{}, {}, {}, {}}}, {{{}, {}, {}, {}}}, {{{c}}}",
            a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]
        ));
    }
    /// The accumulators to shared memory as a row-major 16×16 f32 tile.
    pub fn wmma_store_f32(&mut self, a: R, acc: &[F; 8], ldm: u32) {
        let c = acc.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", ");
        self.ins(format!("wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [{a}], {{{c}}}, {ldm}"));
    }

    // ── control flow ─────────────────────────────────────────────────────────

    pub fn label(&mut self) -> Label {
        self.labels += 1;
        Label(self.labels - 1)
    }
    pub fn place(&mut self, l: Label) {
        let _ = writeln!(self.body, "{l}:");
    }
    pub fn bra(&mut self, l: Label) {
        self.ins(format!("bra.uni {l}"));
    }
    pub fn bra_if(&mut self, p: P, l: Label) {
        self.ins(format!("@{p} bra {l}"));
    }
    pub fn bra_unless(&mut self, p: P, l: Label) {
        self.ins(format!("@!{p} bra {l}"));
    }
    pub fn ret(&mut self) {
        self.ins("ret");
    }
    /// `if p { body }`.
    pub fn if_(&mut self, p: P, body: impl FnOnce(&mut Kernel)) {
        let end = self.label();
        self.bra_unless(p, end);
        body(self);
        self.place(end);
    }
    /// `for i in (start..end).step_by(step) { body(i) }`, `end` exclusive,
    /// tested before the first pass.
    pub fn for_(&mut self, start: impl Into<U>, end: impl Into<U>, step: u32, body: impl FnOnce(&mut Kernel, R)) {
        let end = end.into();
        let i = self.mov(start);
        let (top, out) = (self.label(), self.label());
        let p = self.setp(Cmp::Ge, i, end);
        self.bra_if(p, out);
        self.place(top);
        body(self, i);
        self.add_to(i, i, step);
        let p = self.setp(Cmp::Lt, i, end);
        self.bra_if(p, top);
        self.place(out);
    }

    // ── the module ───────────────────────────────────────────────────────────

    /// The module's text.
    pub fn finish(self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "//\n// {}, emitted by herbert-cuda-f16\n//\n.version {VERSION}\n.target {TARGET}\n.address_size 64\n", self.name);
        for (sym, bytes) in &self.shared {
            let _ = writeln!(s, ".shared .align 16 .b8 {sym}[{bytes}];");
        }
        if self.dynamic_shared {
            let _ = writeln!(s, ".extern .shared .align 16 .b8 {}_dyn[];", self.name);
        }
        let params: Vec<String> = self
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let t = match p {
                    Param::Ptr => "u64",
                    Param::U32 => "u32",
                    Param::F32 => "f32",
                };
                format!("\t.param .{t} {}_p{i}", self.name)
            })
            .collect();
        let _ = writeln!(s, "\n.visible .entry {}(\n{}\n)\n.reqntid {}, 1, 1\n{{", self.name, params.join(",\n"), self.threads);
        for (class, (decl, prefix)) in [(".pred", "%p"), (".b16", "%h"), (".b32", "%r"), (".f32", "%f"), (".b64", "%rd")].iter().enumerate() {
            if self.next[class] > 0 {
                let _ = writeln!(s, "\t.reg {decl} {prefix}<{}>;", self.next[class]);
            }
        }
        s.push('\n');
        s.push_str(&self.body);
        s.push_str("\tret;\n}\n");
        s
    }
}

/// An emitted kernel: its entry name (the module holds that one entry), its
/// text, the threads of a block, the dynamic shared bytes of a launch.
#[derive(Clone)]
pub struct Emitted {
    pub name: String,
    pub ptx: String,
    pub threads: u32,
    pub dyn_shared: u32,
}

impl Emitted {
    pub fn of(k: Kernel, dyn_shared: u32) -> Emitted {
        let (name, threads) = (k.name().to_string(), k.threads());
        Emitted { name, threads, ptx: k.finish(), dyn_shared }
    }

    /// Write `<dir>/<name>.ptx`.
    pub fn dump(&self, dir: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(format!("{}.ptx", self.name)), &self.ptx)
    }
}
