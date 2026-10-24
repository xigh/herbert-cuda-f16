//! One NVIDIA device through the driver: its primary context, one stream,
//! device buffers, modules loaded from emitted PTX, launches, events.
//!
//! What enters: a device ordinal. What comes out: a [`Gpu`] that allocates,
//! copies, loads [`Emitted`] kernels (the driver's ptxas compiles them, its
//! log is returned on failure) and launches them on its stream.
//!
//! Single-threaded use: the context is made current on the thread that opens
//! the device.

use super::driver::{attr, check, drv, fattr, jit, CuContext, CuDevice, CuDevicePtr, CuEvent, CuFunction, CuGraph, CuGraphExec, CuModule, CuStream};
use crate::core::error::{HerbertError, Result};
use crate::ptx::Emitted;
use std::ffi::{c_void, CString};

pub struct Gpu {
    pub device: CuDevice,
    ctx: CuContext,
    pub stream: CuStream,
    pub name: String,
    pub sms: u32,
    pub cc: (u32, u32),
    pub smem_optin: u32,
    pub l2: u32,
    pub driver_version: u32,
}

/// A device allocation, freed on drop.
pub struct Buf {
    pub ptr: CuDevicePtr,
    pub bytes: usize,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            // SAFETY: a pointer `cuMemAlloc` returned, freed once.
            unsafe { (d.mem_free)(self.ptr) };
        }
    }
}

/// Every allocation of an engine, freed together; buffers are addressed by
/// their device pointer.
#[derive(Default)]
pub struct Arena {
    bufs: Vec<Buf>,
    pub bytes: usize,
}

impl Arena {
    pub fn alloc(&mut self, gpu: &Gpu, bytes: usize) -> Result<CuDevicePtr> {
        let b = gpu.alloc(bytes)?;
        gpu.zero(&b)?;
        let p = b.ptr;
        self.bytes += bytes;
        self.bufs.push(b);
        Ok(p)
    }
    pub fn upload<T: Copy>(&mut self, gpu: &Gpu, data: &[T]) -> Result<CuDevicePtr> {
        let b = gpu.upload(data)?;
        let p = b.ptr;
        self.bytes += b.bytes;
        self.bufs.push(b);
        Ok(p)
    }
}

/// A captured sequence of launches, replayed by one call.
pub struct Graph {
    graph: CuGraph,
    exec: CuGraphExec,
}

impl Drop for Graph {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            // SAFETY: a graph and its instance created by `Gpu::capture`, destroyed once.
            unsafe {
                (d.graph_exec_destroy)(self.exec);
                (d.graph_destroy)(self.graph);
            }
        }
    }
}

/// A loaded kernel: its module (kept loaded), its function, what the driver's
/// ptxas made of it.
pub struct Kernel {
    module: CuModule,
    pub func: CuFunction,
    pub name: String,
    pub threads: u32,
    pub dyn_shared: u32,
    pub regs: u32,
    /// Spill bytes per thread (local memory).
    pub local: u32,
    pub static_shared: u32,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            // SAFETY: a module `cuModuleLoadDataEx` returned, unloaded once.
            unsafe { (d.module_unload)(self.module) };
        }
    }
}

/// A kernel argument.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Ptr(CuDevicePtr),
    U32(u32),
    F32(f32),
}

impl From<&Buf> for Arg {
    fn from(b: &Buf) -> Arg {
        Arg::Ptr(b.ptr)
    }
}

/// A GPU timestamp.
pub struct Event(CuEvent);

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            // SAFETY: an event `cuEventCreate` returned, destroyed once.
            unsafe { (d.event_destroy)(self.0) };
        }
    }
}

fn device_attr(dev: CuDevice, a: i32) -> Result<u32> {
    let d = drv()?;
    let mut v = 0;
    // SAFETY: a valid device and attribute; the driver writes one int.
    check(unsafe { (d.device_get_attribute)(&mut v, a, dev) }, "cuDeviceGetAttribute")?;
    Ok(v as u32)
}

impl Gpu {
    /// The devices the driver sees: (ordinal, name).
    pub fn list() -> Result<Vec<(i32, String)>> {
        let d = drv()?;
        let mut n = 0;
        // SAFETY: the driver writes one int.
        check(unsafe { (d.device_get_count)(&mut n) }, "cuDeviceGetCount")?;
        let mut v = Vec::new();
        for i in 0..n {
            let mut dev = 0;
            // SAFETY: an ordinal below the count.
            check(unsafe { (d.device_get)(&mut dev, i) }, "cuDeviceGet")?;
            v.push((i, Self::device_name(dev)?));
        }
        Ok(v)
    }

    fn device_name(dev: CuDevice) -> Result<String> {
        let d = drv()?;
        let mut buf = [0 as std::ffi::c_char; 256];
        // SAFETY: the driver writes a NUL-terminated name within the buffer.
        check(unsafe { (d.device_get_name)(buf.as_mut_ptr(), buf.len() as i32, dev) }, "cuDeviceGetName")?;
        // SAFETY: NUL-terminated by the driver.
        Ok(unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
    }

    /// Device `ordinal`, its primary context made current, one stream.
    pub fn open(ordinal: i32) -> Result<Gpu> {
        let d = drv()?;
        let mut dev = 0;
        // SAFETY: the driver checks the ordinal.
        check(unsafe { (d.device_get)(&mut dev, ordinal) }, "cuDeviceGet")?;
        let mut ctx = std::ptr::null_mut();
        // SAFETY: a valid device.
        check(unsafe { (d.primary_ctx_retain)(&mut ctx, dev) }, "cuDevicePrimaryCtxRetain")?;
        // SAFETY: a context just retained.
        check(unsafe { (d.ctx_set_current)(ctx) }, "cuCtxSetCurrent")?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: a current context. A blocking stream observes the legacy
        // default-stream copies/memsets used by write/zero.
        check(unsafe { (d.stream_create)(&mut stream, 0) }, "cuStreamCreate")?;
        let mut version = 0;
        // SAFETY: the driver writes one int.
        check(unsafe { (d.driver_get_version)(&mut version) }, "cuDriverGetVersion")?;
        Ok(Gpu {
            device: dev,
            ctx,
            stream,
            name: Self::device_name(dev)?,
            sms: device_attr(dev, attr::MULTIPROCESSOR_COUNT)?,
            cc: (device_attr(dev, attr::COMPUTE_CAPABILITY_MAJOR)?, device_attr(dev, attr::COMPUTE_CAPABILITY_MINOR)?),
            smem_optin: device_attr(dev, attr::MAX_SHARED_PER_BLOCK_OPTIN)?,
            l2: device_attr(dev, attr::L2_CACHE_SIZE)?,
            driver_version: version as u32,
        })
    }

    pub fn attr(&self, a: i32) -> Result<u32> {
        device_attr(self.device, a)
    }

    // ── memory ───────────────────────────────────────────────────────────────

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        let d = drv()?;
        let mut ptr = 0;
        // SAFETY: a current context; zero bytes is refused by the driver.
        check(unsafe { (d.mem_alloc)(&mut ptr, bytes.max(4)) }, &format!("cuMemAlloc({bytes})"))?;
        Ok(Buf { ptr, bytes })
    }

    /// A buffer holding `data`.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buf> {
        let bytes = crate::core::bf16::as_bytes(data);
        let b = self.alloc(bytes.len())?;
        self.write(&b, 0, bytes)?;
        Ok(b)
    }

    /// Copy `data` into `b` at byte `at` (synchronous).
    pub fn write(&self, b: &Buf, at: usize, data: &[u8]) -> Result<()> {
        if at + data.len() > b.bytes {
            return Err(HerbertError::Backend(format!("write of {} bytes at {at} into a buffer of {}", data.len(), b.bytes)));
        }
        let d = drv()?;
        // SAFETY: bounds checked above; the host slice is valid for its length.
        check(unsafe { (d.memcpy_htod)(b.ptr + at as u64, data.as_ptr() as *const c_void, data.len()) }, "cuMemcpyHtoD")
    }

    /// Read `out.len()` elements of `b` from byte `at` (waits for the stream).
    pub fn read<T: Copy>(&self, b: &Buf, at: usize, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        if at + bytes > b.bytes {
            return Err(HerbertError::Backend(format!("read of {bytes} bytes at {at} from a buffer of {}", b.bytes)));
        }
        self.sync()?;
        let d = drv()?;
        // SAFETY: bounds checked above; `out` is plain numbers, any byte pattern valid.
        check(unsafe { (d.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, b.ptr + at as u64, bytes) }, "cuMemcpyDtoH")
    }

    pub fn zero(&self, b: &Buf) -> Result<()> {
        let d = drv()?;
        // SAFETY: the whole allocation.
        check(unsafe { (d.memset_d8)(b.ptr, 0, b.bytes) }, "cuMemsetD8")
    }

    /// (free, total) bytes of device memory.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let d = drv()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: the driver writes two sizes.
        check(unsafe { (d.mem_get_info)(&mut free, &mut total) }, "cuMemGetInfo")?;
        Ok((free, total))
    }

    // ── kernels ──────────────────────────────────────────────────────────────

    /// Compile and load an emitted kernel; the ptxas log on failure.
    pub fn load(&self, e: &Emitted) -> Result<Kernel> {
        let d = drv()?;
        let text = CString::new(for_device(&e.ptx, self.cc).map_err(|m| HerbertError::Backend(format!("{}: {m}", e.name)))?)
            .map_err(|_| HerbertError::Backend(format!("{}: NUL in the PTX", e.name)))?;
        let mut info = vec![0u8; 1 << 14];
        let mut error = vec![0u8; 1 << 14];
        let mut opts = [jit::INFO_LOG_BUFFER, jit::INFO_LOG_BUFFER_SIZE_BYTES, jit::ERROR_LOG_BUFFER, jit::ERROR_LOG_BUFFER_SIZE_BYTES, jit::LOG_VERBOSE];
        let mut vals: [*mut c_void; 5] = [info.as_mut_ptr() as *mut c_void, info.len() as *mut c_void, error.as_mut_ptr() as *mut c_void, error.len() as *mut c_void, 1usize as *mut c_void];
        let mut module = std::ptr::null_mut();
        // SAFETY: NUL-terminated PTX; option values as `cuda.h` documents them
        // (sizes passed by value in the pointer); the log buffers outlive the call.
        let r = unsafe { (d.module_load_data_ex)(&mut module, text.as_ptr() as *const c_void, opts.len() as u32, opts.as_mut_ptr(), vals.as_mut_ptr()) };
        if r != 0 {
            let log = String::from_utf8_lossy(&error[..error.iter().position(|&c| c == 0).unwrap_or(0)]).into_owned();
            return Err(check(r, &format!("cuModuleLoadDataEx({})", e.name)).err().map(|x| HerbertError::Backend(format!("{x}\n{log}"))).unwrap());
        }
        let mut func = std::ptr::null_mut();
        let cname = CString::new(e.name.as_str()).unwrap();
        // SAFETY: a loaded module and a NUL-terminated entry name.
        check(unsafe { (d.module_get_function)(&mut func, module, cname.as_ptr()) }, &format!("cuModuleGetFunction({})", e.name))?;
        let fa = |a: i32| -> Result<u32> {
            let mut v = 0;
            // SAFETY: a valid function and attribute.
            check(unsafe { (d.func_get_attribute)(&mut v, a, func) }, "cuFuncGetAttribute")?;
            Ok(v as u32)
        };
        let k = Kernel {
            module,
            func,
            name: e.name.clone(),
            threads: e.threads,
            dyn_shared: e.dyn_shared,
            regs: fa(fattr::NUM_REGS)?,
            local: fa(fattr::LOCAL_SIZE_BYTES)?,
            static_shared: fa(fattr::SHARED_SIZE_BYTES)?,
        };
        if e.dyn_shared > 48 * 1024 {
            // SAFETY: a valid function; the size is checked by the driver.
            check(unsafe { (d.func_set_attribute)(func, fattr::MAX_DYNAMIC_SHARED_SIZE_BYTES, e.dyn_shared as i32) }, &format!("cuFuncSetAttribute({})", e.name))?;
        }
        Ok(k)
    }

    /// Preferred shared/L1 split, in percent. A hint; the driver may override it.
    pub fn set_shared_carveout(&self, k: &Kernel, percent: u32) -> Result<()> {
        if percent > 100 { return Err(HerbertError::Config("shared carveout must be 0..100".into())); }
        // SAFETY: the kernel belongs to this current context, value is in range.
        check(unsafe { (drv()?.func_set_attribute)(k.func, fattr::PREFERRED_SHARED_MEMORY_CARVEOUT, percent as i32) }, "cuFuncSetAttribute(shared carveout)")
    }

    /// Driver resource bound, not sampled/achieved occupancy on a real launch.
    pub fn launch_resources(&self, k: &Kernel) -> Result<String> {
        let d = drv()?;
        let mut blocks = 0;
        let mut preferred = 0;
        // SAFETY: valid function and output pointers in its current context.
        check(unsafe { (d.occupancy_max_active_blocks)(&mut blocks, k.func, k.threads as i32, k.dyn_shared as usize) }, "cuOccupancyMaxActiveBlocksPerMultiprocessor")?;
        check(unsafe { (d.func_get_attribute)(&mut preferred, fattr::PREFERRED_SHARED_MEMORY_CARVEOUT, k.func) }, "cuFuncGetAttribute(shared carveout)")?;
        Ok(format!("{} regs, {} B dynamic shared, {} B local, preferred carveout {}, driver bound {} CTA/SM", k.regs, k.dyn_shared, k.local, preferred, blocks))
    }

    /// Launch `k` on the stream over `grid` blocks.
    pub fn launch(&self, k: &Kernel, grid: (u32, u32, u32), args: &[Arg]) -> Result<()> {
        let d = drv()?;
        let mut storage: Vec<u64> = args
            .iter()
            .map(|a| match *a {
                Arg::Ptr(p) => p,
                Arg::U32(v) => v as u64,
                Arg::F32(v) => v.to_bits() as u64,
            })
            .collect();
        // Each parameter's value at the start of its 8-byte slot (little-endian:
        // a u32 or f32 parameter reads the low 4 bytes).
        let mut ptrs: Vec<*mut c_void> = storage.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
        // SAFETY: the parameter pointers live until the call returns (the
        // driver copies the values); the shapes match the kernel's `.param`s.
        check(
            unsafe { (d.launch_kernel)(k.func, grid.0, grid.1, grid.2, k.threads, 1, 1, k.dyn_shared, self.stream, ptrs.as_mut_ptr(), std::ptr::null_mut()) },
            &format!("cuLaunchKernel({})", k.name),
        )
    }

    /// Capture the launches `record` makes on the stream into a graph.
    pub fn capture(&self, record: impl FnOnce() -> Result<()>) -> Result<Graph> {
        let d = drv()?;
        // SAFETY: the stream of this context; mode 1 = thread-local capture.
        check(unsafe { (d.stream_begin_capture)(self.stream, 1) }, "cuStreamBeginCapture")?;
        let recorded = record();
        let mut graph = std::ptr::null_mut();
        // SAFETY: ends the capture begun above, whatever `record` did.
        let ended = check(unsafe { (d.stream_end_capture)(self.stream, &mut graph) }, "cuStreamEndCapture");
        recorded?;
        ended?;
        let mut exec = std::ptr::null_mut();
        // SAFETY: a graph just captured.
        check(unsafe { (d.graph_instantiate)(&mut exec, graph, 0) }, "cuGraphInstantiateWithFlags")?;
        Ok(Graph { graph, exec })
    }

    pub fn replay(&self, g: &Graph) -> Result<()> {
        let d = drv()?;
        // SAFETY: an instantiated graph of this context, on its stream.
        check(unsafe { (d.graph_launch)(g.exec, self.stream) }, "cuGraphLaunch")
    }

    /// Copy `data` to the device at `ptr` (synchronous).
    pub fn write_raw(&self, ptr: CuDevicePtr, data: &[u8]) -> Result<()> {
        let d = drv()?;
        // SAFETY: the caller passes a pointer into a live allocation large enough.
        check(unsafe { (d.memcpy_htod)(ptr, data.as_ptr() as *const c_void, data.len()) }, "cuMemcpyHtoD")
    }

    /// Read `out.len()` elements at `ptr` (waits for the stream).
    pub fn read_raw<T: Copy>(&self, ptr: CuDevicePtr, out: &mut [T]) -> Result<()> {
        self.sync()?;
        let d = drv()?;
        // SAFETY: the caller passes a pointer into a live allocation large enough.
        check(unsafe { (d.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, ptr, std::mem::size_of_val(out)) }, "cuMemcpyDtoH")
    }

    pub fn zero_raw(&self, ptr: CuDevicePtr, bytes: usize) -> Result<()> {
        let d = drv()?;
        // SAFETY: the caller passes a live allocation of at least `bytes`.
        check(unsafe { (d.memset_d8)(ptr, 0, bytes) }, "cuMemsetD8")
    }

    pub fn sync(&self) -> Result<()> {
        let d = drv()?;
        // SAFETY: the stream of this context.
        check(unsafe { (d.stream_synchronize)(self.stream) }, "cuStreamSynchronize")
    }

    // ── time ─────────────────────────────────────────────────────────────────

    pub fn event(&self) -> Result<Event> {
        let d = drv()?;
        let mut e = std::ptr::null_mut();
        // SAFETY: a current context; default flags (timing on).
        check(unsafe { (d.event_create)(&mut e, 0) }, "cuEventCreate")?;
        Ok(Event(e))
    }

    pub fn record(&self, e: &Event) -> Result<()> {
        let d = drv()?;
        // SAFETY: an event and the stream of this context.
        check(unsafe { (d.event_record)(e.0, self.stream) }, "cuEventRecord")
    }

    /// Milliseconds between two recorded events (waits for the second).
    pub fn elapsed(&self, a: &Event, b: &Event) -> Result<f64> {
        let d = drv()?;
        // SAFETY: a recorded event.
        check(unsafe { (d.event_synchronize)(b.0) }, "cuEventSynchronize")?;
        let mut ms = 0f32;
        // SAFETY: two recorded, completed events.
        check(unsafe { (d.event_elapsed_time)(&mut ms, a.0, b.0) }, "cuEventElapsedTime")?;
        Ok(ms as f64)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            // SAFETY: the stream and the retained primary context of this Gpu.
            unsafe {
                (d.stream_synchronize)(self.stream);
                (d.stream_destroy)(self.stream);
                let _ = self.ctx;
                (d.primary_ctx_release)(self.device);
            }
        }
    }
}

/// The emitted PTX targets sm_89 (`ptx::TARGET`); the driver compiles it
/// forward for newer GPUs, not backward. On sm_80 … sm_88 the target line is
/// lowered to the device's: the kernels use nothing past sm_80 (bf16 and its
/// `mma`, `ldmatrix`, `cp.async`, mbarriers), which `ptxas` confirms for
/// sm_80 and sm_86. Below sm_80 there is no bf16 `mma`.
pub fn for_device(ptx: &str, cc: (u32, u32)) -> std::result::Result<String, String> {
    let target = format!(".target {}\n", crate::ptx::TARGET);
    if cc >= (8, 9) {
        Ok(ptx.to_string())
    } else if cc >= (8, 0) {
        if !ptx.contains(&target) {
            return Err(format!("no `{}` line to lower", target.trim_end()));
        }
        Ok(ptx.replacen(&target, &format!(".target sm_{}{}\n", cc.0, cc.1), 1))
    } else {
        Err(format!("sm_{}{}: the bf16 tensor cores need sm_80 or later", cc.0, cc.1))
    }
}
