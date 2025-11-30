#include "bnb_jax/ffi_handlers.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace bnb_jax {
template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
        "Encapsulated function must be an XLA FFI handler");
    return nb::capsule(reinterpret_cast<const void *>(fn));
}

// Register handler with XLA FFI binding
XLA_FFI_DEFINE_HANDLER(Dequantize_NF4_FP16, Dequantize_NF4_FP16_Impl,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("blocksize")
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::U8>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Ret<xla::ffi::Buffer<xla::ffi::DataType::F16>>());

XLA_FFI_DEFINE_HANDLER(Dequantize_NF4_BF16, Dequantize_NF4_BF16_Impl,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("blocksize")
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::U8>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Ret<xla::ffi::Buffer<xla::ffi::DataType::BF16>>());

XLA_FFI_DEFINE_HANDLER(Dequantize_NF4_FP32, Dequantize_NF4_FP32_Impl,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("blocksize")
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::U8>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Ret<xla::ffi::Buffer<xla::ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER(Dequantize_FP16, Dequantize_FP16_Impl,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("blocksize")
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::U8>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Ret<xla::ffi::Buffer<xla::ffi::DataType::F16>>());

XLA_FFI_DEFINE_HANDLER(Dequantize_BF16, Dequantize_BF16_Impl,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("blocksize")
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::U8>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Ret<xla::ffi::Buffer<xla::ffi::DataType::BF16>>());

XLA_FFI_DEFINE_HANDLER(Dequantize_FP32, Dequantize_FP32_Impl,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int64_t>("blocksize")
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::U8>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Arg<xla::ffi::Buffer<xla::ffi::DataType::F32>>()
                           .Ret<xla::ffi::Buffer<xla::ffi::DataType::F32>>());

nb::dict FFIRegistrations(uintptr_t handle) {
    initialize_from_handle(reinterpret_cast<void *>(handle));

    nb::dict registrations;
    registrations["bnb_dequant_nf4_fp16"] =
        EncapsulateFfiCall(Dequantize_NF4_FP16);
    registrations["bnb_dequant_nf4_bf16"] =
        EncapsulateFfiCall(Dequantize_NF4_BF16);
    registrations["bnb_dequant_nf4_fp32"] =
        EncapsulateFfiCall(Dequantize_NF4_FP32);
    registrations["bnb_dequant_fp16"] =
        EncapsulateFfiCall(Dequantize_FP16);
    registrations["bnb_dequant_bf16"] =
        EncapsulateFfiCall(Dequantize_BF16);
    registrations["bnb_dequant_fp32"] =
        EncapsulateFfiCall(Dequantize_FP32);
    return registrations;
}

} // namespace bnb_jax

NB_MODULE(_core, m) {
    m.doc() = "bitsandbytes-jax native module";

    m.def("ffi_registrations", &bnb_jax::FFIRegistrations, nb::arg("handle"),
          "Get FFI registrations from bitsandbytes library handle (from ctypes "
          "_handle)");
}
