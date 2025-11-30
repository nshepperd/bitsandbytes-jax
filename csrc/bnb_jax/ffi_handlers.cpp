#include "bnb_jax/ffi_handlers.h"

#include <dlfcn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "check.h"

namespace bnb_jax {

// Function pointer type matching bitsandbytes cdequantize_blockwise_fp16_nf4 signature
template<typename T>
using cdequantize_blockwise_t = void (*)(
    float* code,           // codebook (unused for NF4, can be nullptr)
    unsigned char* A,      // packed 4-bit input
    float* absmax,         // per-block scale factors
    T* out,                // output buffer
    int blocksize,         // quantization block size
    int n,                 // number of output elements
    cudaStream_t stream    // CUDA stream
);

struct bnb_lib {
    cdequantize_blockwise_t<half> cdequantize_blockwise_fp16_nf4 = nullptr;
    cdequantize_blockwise_t<__nv_bfloat16> cdequantize_blockwise_bf16_nf4 = nullptr;
    cdequantize_blockwise_t<float> cdequantize_blockwise_fp32_nf4 = nullptr;

    // 8-bit (codebook) dequant
    cdequantize_blockwise_t<half> cdequantize_blockwise_fp16 = nullptr;
    cdequantize_blockwise_t<__nv_bfloat16> cdequantize_blockwise_bf16 = nullptr;
    cdequantize_blockwise_t<float> cdequantize_blockwise_fp32 = nullptr;

};

bnb_lib* g_bnb_lib() {
    static bnb_lib lib;
    return &lib;
}

bool initialize_from_handle(void* handle) {
    if (handle == nullptr) {
        return false;
    }

    // Clear any existing error
    dlerror();

    bnb_lib* lib = g_bnb_lib();

    lib->cdequantize_blockwise_fp16_nf4 = reinterpret_cast<cdequantize_blockwise_t<half>>(
        dlsym(handle, "cdequantize_blockwise_fp16_nf4")
    );
    lib->cdequantize_blockwise_bf16_nf4 = reinterpret_cast<cdequantize_blockwise_t<__nv_bfloat16>>(
        dlsym(handle, "cdequantize_blockwise_bf16_nf4")
    );
    lib->cdequantize_blockwise_fp32_nf4 = reinterpret_cast<cdequantize_blockwise_t<float>>(
        dlsym(handle, "cdequantize_blockwise_fp32_nf4")
    );
    lib->cdequantize_blockwise_fp16 = reinterpret_cast<cdequantize_blockwise_t<half>>(
        dlsym(handle, "cdequantize_blockwise_fp16")
    );
    lib->cdequantize_blockwise_bf16 = reinterpret_cast<cdequantize_blockwise_t<__nv_bfloat16>>(
        dlsym(handle, "cdequantize_blockwise_bf16")
    );
    lib->cdequantize_blockwise_fp32 = reinterpret_cast<cdequantize_blockwise_t<float>>(
        dlsym(handle, "cdequantize_blockwise_fp32")
    );

    const char* error = dlerror();
    if (error != nullptr) {
        return false;
    }
    return true;
}

// FFI Handler implementation
xla::ffi::Error Dequantize_NF4_FP16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F16>> output
) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_fp16_nf4);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() * 2 == n); // 4-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    g_bnb_lib()->cdequantize_blockwise_fp16_nf4(
        nullptr,                                         // code (unused for NF4)
        const_cast<unsigned char*>(quantized.typed_data()),
        const_cast<float*>(absmax.typed_data()),
        reinterpret_cast<half*>(output->typed_data()),
        static_cast<int>(blocksize),
        static_cast<int>(n),
        stream
    );

    return xla::ffi::Error::Success();
}

xla::ffi::Error Dequantize_NF4_BF16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::BF16>> output
) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_bf16_nf4);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() * 2 == n); // 4-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    g_bnb_lib()->cdequantize_blockwise_bf16_nf4(
        nullptr,                                         // code (unused for NF4)
        const_cast<unsigned char*>(quantized.typed_data()),
        const_cast<float*>(absmax.typed_data()),
        reinterpret_cast<__nv_bfloat16*>(output->typed_data()),
        static_cast<int>(blocksize),
        static_cast<int>(n),
        stream
    );

    return xla::ffi::Error::Success();
}

xla::ffi::Error Dequantize_NF4_FP32_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> output
) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_fp32_nf4);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() * 2 == n); // 4-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    g_bnb_lib()->cdequantize_blockwise_fp32_nf4(
        nullptr,                                         // code (unused for NF4)
        const_cast<unsigned char*>(quantized.typed_data()),
        const_cast<float*>(absmax.typed_data()),
        reinterpret_cast<float*>(output->typed_data()),
        static_cast<int>(blocksize),
        static_cast<int>(n),
        stream
    );

    return xla::ffi::Error::Success();
}

xla::ffi::Error Dequantize_FP16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Buffer<xla::ffi::DataType::F32> code,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F16>> output
) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_fp16);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() == n); // 8-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    g_bnb_lib()->cdequantize_blockwise_fp16(
        code.typed_data(),
        quantized.typed_data(),
        absmax.typed_data(),
        reinterpret_cast<half*>(output->typed_data()),
        static_cast<int>(blocksize),
        static_cast<int>(n),
        stream
    );

    return xla::ffi::Error::Success();
}

xla::ffi::Error Dequantize_BF16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Buffer<xla::ffi::DataType::F32> code,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::BF16>> output
) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_bf16);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() == n); // 8-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    g_bnb_lib()->cdequantize_blockwise_bf16(
        code.typed_data(),
        quantized.typed_data(),
        absmax.typed_data(),
        reinterpret_cast<__nv_bfloat16*>(output->typed_data()),
        static_cast<int>(blocksize),
        static_cast<int>(n),
        stream
    );

    return xla::ffi::Error::Success();
}

xla::ffi::Error Dequantize_FP32_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Buffer<xla::ffi::DataType::F32> code,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> output
) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_fp32);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() == n); // 8-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    g_bnb_lib()->cdequantize_blockwise_fp32(
        code.typed_data(),
        quantized.typed_data(),
        absmax.typed_data(),
        reinterpret_cast<float*>(output->typed_data()),
        static_cast<int>(blocksize),
        static_cast<int>(n),
        stream
    );

    return xla::ffi::Error::Success();
}

}  // namespace bnb_jax
