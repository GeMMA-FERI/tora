#pragma once
#include <torch/torch.h>
#include "Properties.hpp"
#include "TensorImpl.hpp"
#include <optional>
#include <tuple>
#include <stdexcept>

namespace PyTorch {
    namespace NodeNamespaces {
        // These are sub-namespaces under PyTorch.
        // The Node Editor treats them as "groups" of extensions.
        // This helps the Node Editor organize them nicely in the context menus.
        inline constexpr auto Convolution = Core::StringLiteral("PyTorch::Convolution");
    }

    static at::Tensor ToTorchValue(const Data::Tensor& value) {
        return ToTorchTensor(value);
    }

    static Data::Tensor FromTorchValue(const at::Tensor& value) {
        return FromTorchTensor(value);
    }

    static Core::Bool FromTorchValue(const bool& value) {
        return Core::Bool(value);
    }

    static Core::Int64 FromTorchValue(const c10::ScalarType& value) {
        return Core::Int64(value);
    }

    template<typename TValue>
    static auto ToTorchValue(const TValue& value) {
        return value;
    }

    template<typename TValue>
    static auto FromTorchValue(const TValue& value) {
        return value;
    }

    static at::ScalarType ToTorchValue(const Data::Enum& value) {
        return static_cast<at::ScalarType>(value.GetIndex());
    }

    static bool ToTorchValue(const Core::Bool& value) {
        return (bool)value;
    }

    static at::string_view ToTorchValue(const Data::String& value) {
        return std::string_view((const char*)value.data(), value.size());
    }
}
