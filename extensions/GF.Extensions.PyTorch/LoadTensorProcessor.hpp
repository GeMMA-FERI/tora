#pragma once
#include "Helpers.hpp"
#include "SafeTensorLoader.hpp"
#include "generated/TensorLoaderBinding.hpp"

struct LoadTensorProcessor : GF::Extensions::impl::DataProcessorBridgeImpl<LoadTensorProcessor> {
    static constexpr auto name() { return TensorLoaderBinding::name(); }
    static constexpr auto description() { return TensorLoaderBinding::description(); }
    static constexpr auto interface_id() { return Core::interfaces::interface_id_storage<GF::Interfaces::Ext::IDataProcessor>::value; }
    // {3A6A2744-F014-45D9-BB93-3D96C9F6B152}
    static constexpr auto implementation_id() { return TensorLoaderBinding::implementation_id(); }
    static auto create_instance() { return Core::make_base_obj<LoadTensorProcessor>().as<Core::Object>(); }
    static Data::Properties create_metadata() {
        Data::Properties metadata;
        AddNamespace<"PyTorch::IO">(metadata);
        return metadata;
    }
    using In_file = TensorLoaderBinding::In_file;
    using In_name = TensorLoaderBinding::In_name;
    using In_base = TensorLoaderBinding::In_base;
    using Out_tensor = TensorLoaderBinding::Out_tensor;

    LoadTensorProcessor() {
        RegisterInput<In_file>();
        RegisterInput<In_name>();
        RegisterOptionalInput<In_base>();
        RegisterOutput<Out_tensor>();
    }
    void Process(const Data::Properties& inputs, Data::Properties outputs) {
        const auto file = std::string(inputs.Get<In_file>());
        const auto name = std::string(inputs.Get<In_name>());
        const auto base = inputs.ContainsValue<In_base>() ? std::string(inputs.Get<In_base>()) : std::string();
        outputs.Set<Out_tensor>(PyTorch::FromTorchTensor(PyTorch::LoadSafeTensor(PyTorch::ResolveTensorFile(file, base), name)));
    }
};
