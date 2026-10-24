//! The CUDA driver API, loaded at run time from `libcuda.so.1` (`nvcuda.dll`
//! on Windows): the handful of entry points the engine calls, no SDK at build
//! time (`cargo check` passes on the Mac).
//!
//! The symbols are the versioned ones the driver exports (`cuCtxCreate_v2`,
//! `cuMemAlloc_v2`…), the ones `cuda.h` maps the plain names to. The table is
//! loaded once ([`drv`]) and lives as long as the process.

use crate::core::error::{HerbertError, Result};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::OnceLock;

/// The driver library: the same entry points under both names.
#[cfg(windows)]
pub const LIBRARY: &str = "nvcuda.dll";
#[cfg(not(windows))]
pub const LIBRARY: &str = "libcuda.so.1";

pub type CuResult = c_int;
pub type CuDevice = c_int;
pub type CuDevicePtr = u64;
pub type CuContext = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuEvent = *mut c_void;
pub type CuGraph = *mut c_void;
pub type CuGraphExec = *mut c_void;

/// `CUdevice_attribute` values the probe reads.
pub mod attr {
    pub const MAX_THREADS_PER_BLOCK: i32 = 1;
    pub const MAX_SHARED_PER_BLOCK: i32 = 8;
    pub const CLOCK_RATE: i32 = 13;
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const MEMORY_CLOCK_RATE: i32 = 36;
    pub const GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;
    pub const L2_CACHE_SIZE: i32 = 38;
    pub const MAX_THREADS_PER_SM: i32 = 39;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    pub const MAX_SHARED_PER_SM: i32 = 81;
    pub const RESERVED_SHARED_PER_BLOCK: i32 = 111;
    pub const MAX_REGISTERS_PER_SM: i32 = 82;
    pub const MAX_SHARED_PER_BLOCK_OPTIN: i32 = 97;
}

/// `CUfunction_attribute` values.
pub mod fattr {
    pub const SHARED_SIZE_BYTES: i32 = 1;
    pub const LOCAL_SIZE_BYTES: i32 = 3;
    pub const NUM_REGS: i32 = 4;
    pub const MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
    pub const PREFERRED_SHARED_MEMORY_CARVEOUT: i32 = 9;
}

/// `CUjit_option` values.
pub mod jit {
    pub const INFO_LOG_BUFFER: u32 = 3;
    pub const INFO_LOG_BUFFER_SIZE_BYTES: u32 = 4;
    pub const ERROR_LOG_BUFFER: u32 = 5;
    pub const ERROR_LOG_BUFFER_SIZE_BYTES: u32 = 6;
    pub const LOG_VERBOSE: u32 = 12;
}

macro_rules! driver {
    ($($field:ident = $sym:literal ($($arg:ty),*);)*) => {
        /// The driver's entry points.
        #[allow(non_snake_case)]
        pub struct Driver {
            _lib: Library,
            $(pub $field: unsafe extern "C" fn($($arg),*) -> CuResult,)*
        }

        impl Driver {
            fn load() -> Result<Driver> {
                // SAFETY: loading the system's CUDA driver library.
                let lib = unsafe { Library::new(LIBRARY) }
                    .map_err(|e| HerbertError::Backend(format!("cannot load {LIBRARY}: {e}")))?;
                $(
                    // SAFETY: the signature is the one of `cuda.h` for this symbol.
                    let $field = *unsafe { lib.get::<unsafe extern "C" fn($($arg),*) -> CuResult>(concat!($sym, "\0").as_bytes()) }
                        .map_err(|e| HerbertError::Backend(format!("{LIBRARY} has no {}: {e}", $sym)))?;
                )*
                Ok(Driver { _lib: lib, $($field,)* })
            }
        }
    };
}

driver! {
    init = "cuInit" (c_uint);
    driver_get_version = "cuDriverGetVersion" (*mut c_int);
    device_get_count = "cuDeviceGetCount" (*mut c_int);
    device_get = "cuDeviceGet" (*mut CuDevice, c_int);
    device_get_name = "cuDeviceGetName" (*mut c_char, c_int, CuDevice);
    device_get_attribute = "cuDeviceGetAttribute" (*mut c_int, c_int, CuDevice);
    device_total_mem = "cuDeviceTotalMem_v2" (*mut usize, CuDevice);
    primary_ctx_retain = "cuDevicePrimaryCtxRetain" (*mut CuContext, CuDevice);
    primary_ctx_release = "cuDevicePrimaryCtxRelease_v2" (CuDevice);
    ctx_set_current = "cuCtxSetCurrent" (CuContext);
    ctx_synchronize = "cuCtxSynchronize" ();
    mem_alloc = "cuMemAlloc_v2" (*mut CuDevicePtr, usize);
    mem_free = "cuMemFree_v2" (CuDevicePtr);
    mem_get_info = "cuMemGetInfo_v2" (*mut usize, *mut usize);
    memcpy_htod = "cuMemcpyHtoD_v2" (CuDevicePtr, *const c_void, usize);
    memcpy_dtoh = "cuMemcpyDtoH_v2" (*mut c_void, CuDevicePtr, usize);
    memcpy_dtod = "cuMemcpyDtoD_v2" (CuDevicePtr, CuDevicePtr, usize);
    memset_d8 = "cuMemsetD8_v2" (CuDevicePtr, u8, usize);
    module_load_data_ex = "cuModuleLoadDataEx" (*mut CuModule, *const c_void, c_uint, *mut c_uint, *mut *mut c_void);
    module_unload = "cuModuleUnload" (CuModule);
    module_get_function = "cuModuleGetFunction" (*mut CuFunction, CuModule, *const c_char);
    func_get_attribute = "cuFuncGetAttribute" (*mut c_int, c_int, CuFunction);
    func_set_attribute = "cuFuncSetAttribute" (CuFunction, c_int, c_int);
    occupancy_max_active_blocks = "cuOccupancyMaxActiveBlocksPerMultiprocessor" (*mut c_int, CuFunction, c_int, usize);
    launch_kernel = "cuLaunchKernel" (CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CuStream, *mut *mut c_void, *mut *mut c_void);
    stream_create = "cuStreamCreate" (*mut CuStream, c_uint);
    stream_destroy = "cuStreamDestroy_v2" (CuStream);
    stream_synchronize = "cuStreamSynchronize" (CuStream);
    stream_query = "cuStreamQuery" (CuStream);
    event_create = "cuEventCreate" (*mut CuEvent, c_uint);
    event_destroy = "cuEventDestroy_v2" (CuEvent);
    event_record = "cuEventRecord" (CuEvent, CuStream);
    event_synchronize = "cuEventSynchronize" (CuEvent);
    event_query = "cuEventQuery" (CuEvent);
    event_elapsed_time = "cuEventElapsedTime" (*mut f32, CuEvent, CuEvent);
    stream_begin_capture = "cuStreamBeginCapture_v2" (CuStream, c_int);
    stream_end_capture = "cuStreamEndCapture" (CuStream, *mut CuGraph);
    graph_instantiate = "cuGraphInstantiateWithFlags" (*mut CuGraphExec, CuGraph, u64);
    graph_launch = "cuGraphLaunch" (CuGraphExec, CuStream);
    graph_exec_destroy = "cuGraphExecDestroy" (CuGraphExec);
    graph_destroy = "cuGraphDestroy" (CuGraph);
    get_error_string = "cuGetErrorString" (CuResult, *mut *const c_char);
}

// SAFETY: the table holds function pointers into a library loaded for the
// whole process; the driver API is thread-safe.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: OnceLock<std::result::Result<Driver, String>> = OnceLock::new();

/// The driver, loaded and initialized on first use.
pub fn drv() -> Result<&'static Driver> {
    let d = DRIVER.get_or_init(|| {
        let d = Driver::load().map_err(|e| e.to_string())?;
        // SAFETY: first call into the driver, flags must be 0.
        let r = unsafe { (d.init)(0) };
        if r != 0 {
            return Err(format!("cuInit: {}", describe(&d, r)));
        }
        Ok(d)
    });
    d.as_ref().map_err(|e| HerbertError::Backend(e.clone()))
}

fn describe(d: &Driver, r: CuResult) -> String {
    let mut s: *const c_char = std::ptr::null();
    // SAFETY: the driver writes a pointer to a static string, or leaves it null.
    unsafe { (d.get_error_string)(r, &mut s) };
    if s.is_null() {
        format!("CUDA error {r}")
    } else {
        // SAFETY: a NUL-terminated static string of the driver.
        format!("{} ({r})", unsafe { CStr::from_ptr(s) }.to_string_lossy())
    }
}

/// A driver call's result, as the crate's error.
pub fn check(r: CuResult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let msg = match drv() {
        Ok(d) => describe(d, r),
        Err(_) => format!("CUDA error {r}"),
    };
    Err(HerbertError::Backend(format!("{what}: {msg}")))
}
