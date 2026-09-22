#pragma once
#include "Helpers.hpp"
#include "SafeTensorLoader.hpp"

struct LoadTensorProcessor : GF::Extensions::impl::DataProcessorBridgeImpl<LoadTensorProcessor> {
    static constexpr auto name() { return std::string_view("Load Tensor"); }
    static constexpr auto description() { return std::string_view("Loads a named tensor from a Safetensors file into owned CPU memory."); }
    static constexpr auto interface_id() { return Core::interfaces::interface_id_storage<GF::Interfaces::Ext::IDataProcessor>::value; }
    // {3A6A2744-F014-45D9-BB93-3D96C9F6B152}
    static constexpr auto implementation_id() { return Core::Uuid(0x3a6a2744, 0xf014, 0x45d9, { 0xbb, 0x93, 0x3d, 0x96, 0xc9, 0xf6, 0xb1, 0x52 }); }
    static auto create_instance() { return Core::make_base_obj<LoadTensorProcessor>().as<Core::Object>(); }
    static Data::Properties create_metadata() {
        Data::Properties metadata;
        AddNamespace<"PyTorch::IO">(metadata);
        return metadata;
    }
    using In_file = Data::FilePathProperty<"File", "Safetensors file. Relative paths require Base Directory.">;
    using In_name = Data::StringProperty<"Tensor Name", "Exact tensor key in the file, for example linear.weight.">;
    using In_base = Data::StringProperty<"Base Directory", "Optional absolute directory used to resolve a relative File.">;
    using Out_tensor = Data::TensorProperty<"Tensor", "Loaded contiguous CPU tensor with its original shape and dtype.">;

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
