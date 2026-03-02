#ifndef BNB_JAX_FFI_HANDLERS_H_
#define BNB_JAX_FFI_HANDLERS_H_

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace bnb_jax {

// Initialize function pointers from a dlopen handle to bitsandbytes library
bool initialize_from_handle(void *handle);

// FFI handler for NF4 dequantization to float16
// XLA_FFI_Error* Dequantize_NF4_FP16(XLA_FFI_CallFrame* call_frame);
xla::ffi::Error
Dequantize_NF4_Impl(cudaStream_t stream, int64_t blocksize,
                    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
                    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
                    xla::ffi::Result<xla::ffi::AnyBuffer> output);

xla::ffi::Error
Dequantize_Blockwise_Impl(cudaStream_t stream, int64_t blocksize,
                          xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
                          xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
                          xla::ffi::Buffer<xla::ffi::DataType::F32> code,
                          xla::ffi::Result<xla::ffi::AnyBuffer> output);

// Quantization handlers
xla::ffi::Error
Quantize_NF4_Impl(cudaStream_t stream, int64_t blocksize,
                  xla::ffi::AnyBuffer input,
                  xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U8>> quantized,
                  xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> absmax);

xla::ffi::Error
Quantize_Blockwise_Impl(cudaStream_t stream, int64_t blocksize,
                        xla::ffi::AnyBuffer input,
                        xla::ffi::Buffer<xla::ffi::DataType::F32> code,
                        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U8>> quantized,
                        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> absmax);

} // namespace bnb_jax

#endif // BNB_JAX_FFI_HANDLERS_H_
