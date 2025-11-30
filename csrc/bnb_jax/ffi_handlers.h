#ifndef BNB_JAX_FFI_HANDLERS_H_
#define BNB_JAX_FFI_HANDLERS_H_

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace bnb_jax {

// Initialize function pointers from a dlopen handle to bitsandbytes library
bool initialize_from_handle(void* handle);

// FFI handler for NF4 dequantization to float16
// XLA_FFI_Error* Dequantize_NF4_FP16(XLA_FFI_CallFrame* call_frame);
xla::ffi::Error Dequantize_NF4_FP16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F16>> output
);

xla::ffi::Error Dequantize_NF4_BF16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::BF16>> output
);

xla::ffi::Error Dequantize_NF4_FP32_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> output
);

xla::ffi::Error Dequantize_FP16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Buffer<xla::ffi::DataType::F32> code,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F16>> output
);

xla::ffi::Error Dequantize_BF16_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Buffer<xla::ffi::DataType::F32> code,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::BF16>> output
);

xla::ffi::Error Dequantize_FP32_Impl(
    cudaStream_t stream,
    int64_t blocksize,
    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
    xla::ffi::Buffer<xla::ffi::DataType::F32> code,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> output
);

}  // namespace bnb_jax

#endif  // BNB_JAX_FFI_HANDLERS_H_
