#pragma once

#include <Data/Tensor.hpp>
#include <Data/TypedArray.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace PyTorch
{
    namespace detail
    {
        inline Data::Structures::TensorDataType FromTorchDataType(at::ScalarType type)
        {
            using Code = Data::Structures::TensorDataTypeCode;

            switch (type) {
                case at::ScalarType::Bool:          return { Code::Bool, 8, 1 };
                case at::ScalarType::Char:          return { Code::Int, 8, 1 };
                case at::ScalarType::Short:         return { Code::Int, 16, 1 };
                case at::ScalarType::Int:           return { Code::Int, 32, 1 };
                case at::ScalarType::Long:          return { Code::Int, 64, 1 };
                case at::ScalarType::Byte:          return { Code::UInt, 8, 1 };
                case at::ScalarType::UInt16:        return { Code::UInt, 16, 1 };
                case at::ScalarType::UInt32:        return { Code::UInt, 32, 1 };
                case at::ScalarType::UInt64:        return { Code::UInt, 64, 1 };
                case at::ScalarType::Half:          return { Code::Float, 16, 1 };
                case at::ScalarType::Float:         return { Code::Float, 32, 1 };
                case at::ScalarType::Double:        return { Code::Float, 64, 1 };
                case at::ScalarType::BFloat16:      return { Code::BFloat, 16, 1 };
                case at::ScalarType::ComplexHalf:   return { Code::Complex, 32, 1 };
                case at::ScalarType::ComplexFloat:  return { Code::Complex, 64, 1 };
                case at::ScalarType::ComplexDouble: return { Code::Complex, 128, 1 };
                default:
                    return {
                        Code::Opaque,
                        static_cast<Core::Byte>(std::min<size_t>(
                            255,
                            c10::elementSize(type) * 8)),
                        1
                    };
                }
        }

        inline at::ScalarType ToTorchDataType(
            Data::Structures::TensorDataType const& type)
        {
            using Code = Data::Structures::TensorDataTypeCode;

            Core::throw_if_false(
                type.Lanes == 1,
                "PyTorch tensor element types must have exactly one lane.");

            switch (type.Code) {
                case Code::Bool:
                    if (type.Bits == 8) return at::ScalarType::Bool;
                    break;
                case Code::Int:
                    switch (type.Bits) {
                        case 8:  return at::ScalarType::Char;
                        case 16: return at::ScalarType::Short;
                        case 32: return at::ScalarType::Int;
                        case 64: return at::ScalarType::Long;
                    }
                    break;
                case Code::UInt:
                    switch (type.Bits) {
                        case 8:  return at::ScalarType::Byte;
                        case 16: return at::ScalarType::UInt16;
                        case 32: return at::ScalarType::UInt32;
                        case 64: return at::ScalarType::UInt64;
                    }
                    break;
                case Code::Float:
                    switch (type.Bits) {
                        case 16: return at::ScalarType::Half;
                        case 32: return at::ScalarType::Float;
                        case 64: return at::ScalarType::Double;
                    }
                    break;
                case Code::BFloat:
                    if (type.Bits == 16) return at::ScalarType::BFloat16;
                    break;
                case Code::Complex:
                    switch (type.Bits) {
                        case 32:  return at::ScalarType::ComplexHalf;
                        case 64:  return at::ScalarType::ComplexFloat;
                        case 128: return at::ScalarType::ComplexDouble;
                    }
                    break;
                default:
                    break;
            }

            throw std::runtime_error("The requested tensor data type is not supported by PyTorch.");
        }

        inline Data::Structures::TensorDevice FromTorchDevice(c10::Device device)
        {
            using Type = Data::Structures::TensorDeviceType;
            Type result = Type::Other;

            switch (device.type()) {
                case c10::DeviceType::CPU:    result = Type::CPU; break;
                case c10::DeviceType::CUDA:   result = Type::CUDA; break;
                case c10::DeviceType::HIP:    result = Type::ROCm; break;
                case c10::DeviceType::Vulkan: result = Type::Vulkan; break;
                case c10::DeviceType::Metal:
                case c10::DeviceType::MPS:    result = Type::Metal; break;
                default: break;
            }

            return {
                result,
                device.has_index() ? static_cast<Core::Int32>(device.index()) : 0
            };
        }

        inline c10::Device ToTorchDevice(Data::Structures::TensorDevice const& device)
        {
            using Type = Data::Structures::TensorDeviceType;

            switch (device.Type) {
                case Type::CPU:
                    return c10::Device(c10::DeviceType::CPU);
                case Type::CUDA:
                    Core::throw_if_false(device.Index >= 0, "CUDA device index cannot be negative.");
                    return c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device.Index));
                case Type::ROCm:
                    Core::throw_if_false(device.Index >= 0, "ROCm device index cannot be negative.");
                    return c10::Device(c10::DeviceType::HIP, static_cast<c10::DeviceIndex>(device.Index));
                case Type::Vulkan:
                    return c10::Device(c10::DeviceType::Vulkan);
                case Type::Metal:
                    return c10::Device(c10::DeviceType::MPS);
                default:
                    throw std::runtime_error("The requested tensor device is not supported by PyTorch.");
            }
        }

        inline Data::Structures::TensorLayout FromTorchLayout(at::Layout layout)
        {
            using Layout = Data::Structures::TensorLayout;

            switch (layout) {
                case at::kStrided:   return Layout::Strided;
                case at::kSparse:    return Layout::SparseCOO;
                case at::kSparseCsr: return Layout::SparseCSR;
                case at::kSparseCsc: return Layout::SparseCSC;
                case at::kSparseBsr: return Layout::SparseBSR;
                case at::kSparseBsc: return Layout::SparseBSC;
                default:             return Layout::Opaque;
            }
        }

        inline at::ScalarType TypedArrayDataType(Data::Structures::ValueType type)
        {
            using Type = Data::Structures::ValueType;

            switch (type) {
                case Type::SByte:  return at::ScalarType::Char;
                case Type::Int16:  return at::ScalarType::Short;
                case Type::Int32:  return at::ScalarType::Int;
                case Type::Int64:  return at::ScalarType::Long;
                case Type::Byte:   return at::ScalarType::Byte;
                case Type::UInt16: return at::ScalarType::UInt16;
                case Type::UInt32: return at::ScalarType::UInt32;
                case Type::UInt64: return at::ScalarType::UInt64;
                case Type::Float:  return at::ScalarType::Float;
                case Type::Double: return at::ScalarType::Double;
                case Type::Bool:   return at::ScalarType::Bool;
                default:
                    throw std::runtime_error("Tensor values must be a TypedArray of supported primitive numbers.");
            }
        }

        inline Core::Int64 DimensionAt(Data::TypedArray const& dimensions, int index)
        {
            using Type = Data::Structures::ValueType;

            switch (dimensions.GetValueType()) {
                case Type::SByte:  return dimensions.Get<Core::SByte>(index);
                case Type::Int16:  return dimensions.Get<Core::Int16>(index);
                case Type::Int32:  return dimensions.Get<Core::Int32>(index);
                case Type::Int64:  return dimensions.Get<Core::Int64>(index);
                case Type::Byte:   return dimensions.Get<Core::Byte>(index);
                case Type::UInt16: return dimensions.Get<Core::UInt16>(index);
                case Type::UInt32: return dimensions.Get<Core::UInt32>(index);
                case Type::UInt64: {
                    const auto value = dimensions.Get<Core::UInt64>(index);
                    Core::throw_if_false(
                        value <= static_cast<Core::UInt64>(std::numeric_limits<Core::Int64>::max()),
                        "Tensor dimension is too large.");
                    return static_cast<Core::Int64>(value);
                }
                default:
                    throw std::runtime_error("Tensor dimensions must be a TypedArray of integral values.");
            }
        }
    }

    class TensorImpl final : public Data::Interfaces::ITensor
    {
    public:
        FC_II_IMPLEMENTBASE;

        explicit TensorImpl(at::Tensor tensor) : m_Tensor(std::move(tensor))
        {
            Core::throw_if_false(m_Tensor.defined(), "Cannot wrap an undefined PyTorch tensor.");
        }

        at::Tensor const& GetTorchTensor() const noexcept
        {
            return m_Tensor;
        }

        void _abi_GetDataType(
            Data::Structures::TensorDataType* resultOut,
            Core::IErrorCode** errorOut) const override
        {
            Core::call_and_translate_exception_for_dll_boundary([&] {
                Core::throw_if_invalid(resultOut);
                *resultOut = detail::FromTorchDataType(m_Tensor.scalar_type());
            }, errorOut);
        }

        void _abi_GetDevice(
            Data::Structures::TensorDevice* resultOut,
            Core::IErrorCode** errorOut) const override
        {
            Core::call_and_translate_exception_for_dll_boundary([&] {
                Core::throw_if_invalid(resultOut);
                *resultOut = detail::FromTorchDevice(m_Tensor.device());
            }, errorOut);
        }

        Data::Structures::TensorLayout _abi_GetLayout() const override
        {
            return detail::FromTorchLayout(m_Tensor.layout());
        }

        Data::Structures::TensorFlags _abi_GetFlags() const override
        {
            using Flags = Data::Structures::TensorFlags;
            auto flags = static_cast<Core::UInt64>(Flags::ReadOnly);

            if (m_Tensor.layout() == at::kStrided && m_Tensor.is_contiguous()) {
                flags |= static_cast<Core::UInt64>(Flags::Contiguous);
            }
            if (m_Tensor.device().is_cpu()) {
                flags |= static_cast<Core::UInt64>(Flags::HostVisible);
            }

            return static_cast<Flags>(flags);
        }

        Core::UInt64 _abi_GetRank() const override
        {
            return static_cast<Core::UInt64>(m_Tensor.dim());
        }

        Core::Int64 _abi_GetSize(
            Core::UInt64 dimension,
            Core::IErrorCode** errorOut) const override
        {
            Core::Int64 result = 0;
            Core::call_and_translate_exception_for_dll_boundary([&] {
                Core::throw_if_false(dimension < _abi_GetRank(), "Tensor dimension is out of range.");
                result = static_cast<Core::Int64>(m_Tensor.size(static_cast<int64_t>(dimension)));
            }, errorOut);
            return result;
        }

        bool _abi_HasStrides() const override
        {
            return m_Tensor.layout() == at::kStrided;
        }

        Core::Int64 _abi_GetStride(
            Core::UInt64 dimension,
            Core::IErrorCode** errorOut) const override
        {
            Core::Int64 result = 0;
            Core::call_and_translate_exception_for_dll_boundary([&] {
                Core::throw_if_false(_abi_HasStrides(), "Tensor layout does not expose dense strides.");
                Core::throw_if_false(dimension < _abi_GetRank(), "Tensor dimension is out of range.");
                result = static_cast<Core::Int64>(m_Tensor.stride(static_cast<int64_t>(dimension)));
            }, errorOut);
            return result;
        }

        Core::Int64 _abi_GetStorageOffset(Core::IErrorCode** errorOut) const override
        {
            Core::Int64 result = 0;
            Core::call_and_translate_exception_for_dll_boundary([&] {
                Core::throw_if_false(_abi_HasStrides(), "Tensor layout does not expose a storage offset.");
                result = static_cast<Core::Int64>(m_Tensor.storage_offset());
            }, errorOut);
            return result;
        }

        void _abi_CopyTo(
            Data::Structures::TensorDevice const* deviceIn,
            Data::Structures::TensorDataType const* dataTypeIn,
            Data::Interfaces::ITensor** resultOut,
            Core::IErrorCode** errorOut) const override
        {
            Core::call_and_translate_exception_for_dll_boundary([&] {
                Core::throw_if_invalid(deviceIn);
                Core::throw_if_invalid(resultOut);

                const auto device = detail::ToTorchDevice(*deviceIn);
                const auto dataType = dataTypeIn == nullptr
                    ? m_Tensor.scalar_type()
                    : detail::ToTorchDataType(*dataTypeIn);

                const auto options = m_Tensor.options().device(device).dtype(dataType);
                auto result = m_Tensor.to(options, false, false);
                *resultOut = new TensorImpl(std::move(result));
            }, errorOut);
        }

    private:
        at::Tensor m_Tensor;
    };

    inline Data::Tensor FromTorchTensor(at::Tensor tensor)
    {
        Data::Tensor result = nullptr;
        auto** resultOut = &result;
        *resultOut = new TensorImpl(std::move(tensor));
        return result;
    }

    inline at::Tensor ToTorchTensor(Data::Tensor const& tensor)
    {
        auto* implementation = dynamic_cast<TensorImpl*>(tensor.get_abi());
        Core::throw_if_false(
            implementation != nullptr,
            "The tensor is not backed by the GF.Extensions.PyTorch runtime.");
        return implementation->GetTorchTensor();
    }

    inline Data::Tensor TensorFromTypedArray(
        Data::TypedArray const& values,
        Data::TypedArray const& dimensions)
    {
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(dimensions.Size()));

        Core::UInt64 elementCount = 1;
        for (int i = 0; i < dimensions.Size(); ++i) {
            const Core::Int64 dimension = detail::DimensionAt(dimensions, i);
            Core::throw_if_false(dimension >= 0, "Tensor dimensions cannot be negative.");

            const auto unsignedDimension = static_cast<Core::UInt64>(dimension);
            if (unsignedDimension == 0) {
                elementCount = 0;
            }
            else if (elementCount != 0) {
                Core::throw_if_false(
                    elementCount <= std::numeric_limits<Core::UInt64>::max() / unsignedDimension,
                    "Tensor shape element count overflow.");
                elementCount *= unsignedDimension;
            }

            shape.push_back(static_cast<int64_t>(dimension));
        }

        Core::throw_if_false(
            elementCount == static_cast<Core::UInt64>(values.Size()),
            "Tensor shape element count must match the flat values count.");

        const auto scalarType = detail::TypedArrayDataType(values.GetValueType());
        const auto options = torch::TensorOptions().dtype(scalarType).device(torch::kCPU);

        if (elementCount == 0) {
            return FromTorchTensor(torch::empty(shape, options));
        }

        Core::throw_if_false(
            values.get_abi()->_abi_GetStride() == c10::elementSize(scalarType),
            "TypedArray element stride does not match its declared numeric type.");

        auto tensor = torch::from_blob(
            values.get_abi()->_abi_GetData(),
            shape,
            options).clone();

        return FromTorchTensor(std::move(tensor));
    }
}
