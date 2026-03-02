#include "bnb_jax/ffi_handlers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include "check.h"

namespace bnb_jax {

// Function pointer type matching bitsandbytes cdequantize_blockwise_fp16_nf4
// signature
template <typename T>
using cdequantize_blockwise_t =
    void (*)(float *code,        // codebook (unused for NF4, can be nullptr)
             unsigned char *A,   // packed 4-bit input
             float *absmax,      // per-block scale factors
             T *out,             // output buffer
             int blocksize,      // quantization block size
             int n,              // number of output elements
             cudaStream_t stream // CUDA stream
    );

// Function pointer type matching bitsandbytes cquantize_blockwise_*
// Note: quantize functions do NOT take a cudaStream_t parameter
template <typename T>
using cquantize_blockwise_t =
    void (*)(float *code,        // codebook (unused for NF4, can be nullptr)
             T *A,               // input tensor
             float *absmax,      // per-block scale factors (output)
             unsigned char *out, // quantized output
             int blocksize,      // quantization block size
             int n               // number of input elements
    );

struct bnb_lib {
    cdequantize_blockwise_t<half> cdequantize_blockwise_fp16_nf4 = nullptr;
    cdequantize_blockwise_t<__nv_bfloat16> cdequantize_blockwise_bf16_nf4 =
        nullptr;
    cdequantize_blockwise_t<float> cdequantize_blockwise_fp32_nf4 = nullptr;

    // 8-bit (codebook) dequant
    cdequantize_blockwise_t<half> cdequantize_blockwise_fp16 = nullptr;
    cdequantize_blockwise_t<__nv_bfloat16> cdequantize_blockwise_bf16 = nullptr;
    cdequantize_blockwise_t<float> cdequantize_blockwise_fp32 = nullptr;

    // NF4 quantize
    cquantize_blockwise_t<half> cquantize_blockwise_fp16_nf4 = nullptr;
    cquantize_blockwise_t<__nv_bfloat16> cquantize_blockwise_bf16_nf4 = nullptr;
    cquantize_blockwise_t<float> cquantize_blockwise_fp32_nf4 = nullptr;

    // 8-bit (codebook) quantize
    cquantize_blockwise_t<half> cquantize_blockwise_fp16 = nullptr;
    cquantize_blockwise_t<__nv_bfloat16> cquantize_blockwise_bf16 = nullptr;
    cquantize_blockwise_t<float> cquantize_blockwise_fp32 = nullptr;
};

bnb_lib *g_bnb_lib() {
    static bnb_lib lib;
    return &lib;
}

bool initialize_from_handle(void *handle) {
    if (handle == nullptr) {
        return false;
    }

    // Clear any existing error
    dlerror();

    bnb_lib *lib = g_bnb_lib();

    lib->cdequantize_blockwise_fp16_nf4 =
        reinterpret_cast<cdequantize_blockwise_t<half>>(
            dlsym(handle, "cdequantize_blockwise_fp16_nf4"));
    lib->cdequantize_blockwise_bf16_nf4 =
        reinterpret_cast<cdequantize_blockwise_t<__nv_bfloat16>>(
            dlsym(handle, "cdequantize_blockwise_bf16_nf4"));
    lib->cdequantize_blockwise_fp32_nf4 =
        reinterpret_cast<cdequantize_blockwise_t<float>>(
            dlsym(handle, "cdequantize_blockwise_fp32_nf4"));
    lib->cdequantize_blockwise_fp16 =
        reinterpret_cast<cdequantize_blockwise_t<half>>(
            dlsym(handle, "cdequantize_blockwise_fp16"));
    lib->cdequantize_blockwise_bf16 =
        reinterpret_cast<cdequantize_blockwise_t<__nv_bfloat16>>(
            dlsym(handle, "cdequantize_blockwise_bf16"));
    lib->cdequantize_blockwise_fp32 =
        reinterpret_cast<cdequantize_blockwise_t<float>>(
            dlsym(handle, "cdequantize_blockwise_fp32"));

    // Quantize NF4
    lib->cquantize_blockwise_fp16_nf4 =
        reinterpret_cast<cquantize_blockwise_t<half>>(
            dlsym(handle, "cquantize_blockwise_fp16_nf4"));
    lib->cquantize_blockwise_bf16_nf4 =
        reinterpret_cast<cquantize_blockwise_t<__nv_bfloat16>>(
            dlsym(handle, "cquantize_blockwise_bf16_nf4"));
    lib->cquantize_blockwise_fp32_nf4 =
        reinterpret_cast<cquantize_blockwise_t<float>>(
            dlsym(handle, "cquantize_blockwise_fp32_nf4"));

    // Quantize 8-bit blockwise
    lib->cquantize_blockwise_fp16 =
        reinterpret_cast<cquantize_blockwise_t<half>>(
            dlsym(handle, "cquantize_blockwise_fp16"));
    lib->cquantize_blockwise_bf16 =
        reinterpret_cast<cquantize_blockwise_t<__nv_bfloat16>>(
            dlsym(handle, "cquantize_blockwise_bf16"));
    lib->cquantize_blockwise_fp32 =
        reinterpret_cast<cquantize_blockwise_t<float>>(
            dlsym(handle, "cquantize_blockwise_fp32"));

    const char *error = dlerror();
    if (error != nullptr) {
        return false;
    }
    return true;
}

// FFI Handler implementation
xla::ffi::Error
Dequantize_NF4_Impl(cudaStream_t stream, int64_t blocksize,
                    xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
                    xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
                    xla::ffi::Result<xla::ffi::AnyBuffer> output) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_fp16_nf4);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() * 2 == n); // 4-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    if (output->element_type() == xla::ffi::DataType::F16) {
        g_bnb_lib()->cdequantize_blockwise_fp16_nf4(
            /*code=*/nullptr, quantized.typed_data(), absmax.typed_data(),
            reinterpret_cast<half *>(output->untyped_data()),
            static_cast<int>(blocksize), static_cast<int>(n), stream);
    } else if (output->element_type() == xla::ffi::DataType::BF16) {
        g_bnb_lib()->cdequantize_blockwise_bf16_nf4(
            /*code=*/nullptr, quantized.typed_data(), absmax.typed_data(),
            reinterpret_cast<__nv_bfloat16 *>(output->untyped_data()),
            static_cast<int>(blocksize), static_cast<int>(n), stream);
    } else if (output->element_type() == xla::ffi::DataType::F32) {
        g_bnb_lib()->cdequantize_blockwise_fp32_nf4(
            /*code=*/nullptr, quantized.typed_data(), absmax.typed_data(),
            reinterpret_cast<float *>(output->untyped_data()),
            static_cast<int>(blocksize), static_cast<int>(n), stream);
    } else {
        return xla::ffi::Error::InvalidArgument(
            "Unsupported output data type for Dequantize_NF4");
    }

    FFI_CUDA_CHECK(cudaGetLastError());

    return xla::ffi::Error::Success();
}

xla::ffi::Error
Dequantize_Blockwise_Impl(cudaStream_t stream, int64_t blocksize,
                          xla::ffi::Buffer<xla::ffi::DataType::U8> quantized,
                          xla::ffi::Buffer<xla::ffi::DataType::F32> absmax,
                          xla::ffi::Buffer<xla::ffi::DataType::F32> code,
                          xla::ffi::Result<xla::ffi::AnyBuffer> output) {
    FFI_CHECK(g_bnb_lib()->cdequantize_blockwise_fp16);

    // Get the output buffer and its size
    int64_t n = output->element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized.element_count() == n); // 8-bit packed
    FFI_CHECK(absmax.element_count() * blocksize == n);

    // Call bitsandbytes kernel
    if (output->element_type() == xla::ffi::DataType::F16) {
        g_bnb_lib()->cdequantize_blockwise_fp16(
            code.typed_data(), quantized.typed_data(), absmax.typed_data(),
            reinterpret_cast<half *>(output->untyped_data()),
            static_cast<int>(blocksize), static_cast<int>(n), stream);
    } else if (output->element_type() == xla::ffi::DataType::BF16) {
        g_bnb_lib()->cdequantize_blockwise_bf16(
            code.typed_data(), quantized.typed_data(), absmax.typed_data(),
            reinterpret_cast<__nv_bfloat16 *>(output->untyped_data()),
            static_cast<int>(blocksize), static_cast<int>(n), stream);
    } else if (output->element_type() == xla::ffi::DataType::F32) {
        g_bnb_lib()->cdequantize_blockwise_fp32(
            code.typed_data(), quantized.typed_data(), absmax.typed_data(),
            reinterpret_cast<float *>(output->untyped_data()),
            static_cast<int>(blocksize), static_cast<int>(n), stream);
    } else {
        return xla::ffi::Error::InvalidArgument(
            "Unsupported output data type for Dequantize_Blockwise");
    }

    return xla::ffi::Error::Success();
}

// Quantize NF4 FFI Handler
xla::ffi::Error
Quantize_NF4_Impl(cudaStream_t stream, int64_t blocksize,
                  xla::ffi::AnyBuffer input,
                  xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U8>> quantized,
                  xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> absmax) {
    FFI_CHECK(g_bnb_lib()->cquantize_blockwise_fp16_nf4);

    int64_t n = input.element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized->element_count() * 2 == n); // 4-bit packed
    FFI_CHECK(absmax->element_count() * blocksize == n);

    // bitsandbytes quantize functions don't accept a stream parameter,
    // so synchronize to ensure input data is ready
    FFI_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (input.element_type() == xla::ffi::DataType::F16) {
        g_bnb_lib()->cquantize_blockwise_fp16_nf4(
            /*code=*/nullptr,
            reinterpret_cast<half *>(input.untyped_data()),
            absmax->typed_data(),
            quantized->typed_data(),
            static_cast<int>(blocksize), static_cast<int>(n));
    } else if (input.element_type() == xla::ffi::DataType::BF16) {
        g_bnb_lib()->cquantize_blockwise_bf16_nf4(
            /*code=*/nullptr,
            reinterpret_cast<__nv_bfloat16 *>(input.untyped_data()),
            absmax->typed_data(),
            quantized->typed_data(),
            static_cast<int>(blocksize), static_cast<int>(n));
    } else if (input.element_type() == xla::ffi::DataType::F32) {
        g_bnb_lib()->cquantize_blockwise_fp32_nf4(
            /*code=*/nullptr,
            reinterpret_cast<float *>(input.untyped_data()),
            absmax->typed_data(),
            quantized->typed_data(),
            static_cast<int>(blocksize), static_cast<int>(n));
    } else {
        return xla::ffi::Error::InvalidArgument(
            "Unsupported input data type for Quantize_NF4");
    }

    // Synchronize default stream to ensure quantization is complete
    // before XLA continues on its stream
    FFI_CUDA_CHECK(cudaDeviceSynchronize());

    return xla::ffi::Error::Success();
}

// Quantize Blockwise (8-bit) FFI Handler
xla::ffi::Error
Quantize_Blockwise_Impl(cudaStream_t stream, int64_t blocksize,
                        xla::ffi::AnyBuffer input,
                        xla::ffi::Buffer<xla::ffi::DataType::F32> code,
                        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U8>> quantized,
                        xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> absmax) {
    FFI_CHECK(g_bnb_lib()->cquantize_blockwise_fp16);

    int64_t n = input.element_count();
    FFI_CHECK(n % blocksize == 0);
    FFI_CHECK(quantized->element_count() == n); // 8-bit: 1 byte per element
    FFI_CHECK(absmax->element_count() * blocksize == n);

    FFI_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (input.element_type() == xla::ffi::DataType::F16) {
        g_bnb_lib()->cquantize_blockwise_fp16(
            code.typed_data(),
            reinterpret_cast<half *>(input.untyped_data()),
            absmax->typed_data(),
            quantized->typed_data(),
            static_cast<int>(blocksize), static_cast<int>(n));
    } else if (input.element_type() == xla::ffi::DataType::BF16) {
        g_bnb_lib()->cquantize_blockwise_bf16(
            code.typed_data(),
            reinterpret_cast<__nv_bfloat16 *>(input.untyped_data()),
            absmax->typed_data(),
            quantized->typed_data(),
            static_cast<int>(blocksize), static_cast<int>(n));
    } else if (input.element_type() == xla::ffi::DataType::F32) {
        g_bnb_lib()->cquantize_blockwise_fp32(
            code.typed_data(),
            reinterpret_cast<float *>(input.untyped_data()),
            absmax->typed_data(),
            quantized->typed_data(),
            static_cast<int>(blocksize), static_cast<int>(n));
    } else {
        return xla::ffi::Error::InvalidArgument(
            "Unsupported input data type for Quantize_Blockwise");
    }

    FFI_CUDA_CHECK(cudaDeviceSynchronize());

    return xla::ffi::Error::Success();
}

} // namespace bnb_jax
