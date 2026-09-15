#pragma once

#include "Helpers.hpp"

struct TensorFromTextProcessor : GF::Extensions::impl::DataProcessorBridgeImpl<TensorFromTextProcessor> {
    static constexpr auto name() { return std::string_view("Tensor From Text"); }
    static constexpr auto description() { return std::string_view("Creates a Float64 CPU tensor from comma- or whitespace-separated values and dimensions."); }
    static constexpr auto interface_id() { return Core::interfaces::interface_id_storage<GF::Interfaces::Ext::IDataProcessor>::value; }

    // {8D9BE809-5B45-4C53-91ED-B7923B005317}
    static constexpr auto implementation_id()
    {
        return Core::Uuid(0x8d9be809, 0x5b45, 0x4c53, { 0x91, 0xed, 0xb7, 0x92, 0x3b, 0x00, 0x53, 0x17 });
    }
    static auto create_instance()
    {
        return Core::make_base_obj<TensorFromTextProcessor>().as<Core::Object>();
    }

    using In_values = Data::StringProperty<"Values", "Comma- or whitespace-separated floating-point values.">;
    using In_dimensions = Data::StringProperty<"Dimensions", "Comma- or whitespace-separated integral dimensions.">;
    using Out_tensor = Data::TensorProperty<"Tensor", "Created Float64 CPU tensor.">;

    TensorFromTextProcessor()
    {
        RegisterInput<In_values>();
        RegisterInput<In_dimensions>();
        RegisterOutput<Out_tensor>();
    }

    void Process(Data::Properties const& inputs, Data::Properties outputs)
    {
        outputs.Set<Out_tensor>(PyTorch::TensorFromText(inputs.Get<In_values>(), inputs.Get<In_dimensions>()));
    }
};

struct TensorToTextProcessor : GF::Extensions::impl::DataProcessorBridgeImpl<TensorToTextProcessor>
{
    static constexpr auto name() { return std::string_view("Tensor To Text"); }
    static constexpr auto description() { return std::string_view("Copies a tensor to CPU when necessary and formats its values, shape and dtype for inspection."); }
    static constexpr auto interface_id() { return Core::interfaces::interface_id_storage<GF::Interfaces::Ext::IDataProcessor>::value; }

    // {D420B414-433B-46F4-84D7-A986A4F27E86}
    static constexpr auto implementation_id() {
        return Core::Uuid(0xd420b414, 0x433b, 0x46f4, { 0x84, 0xd7, 0xa9, 0x86, 0xa4, 0xf2, 0x7e, 0x86 });
    }
    static auto create_instance()
    {
        return Core::make_base_obj<TensorToTextProcessor>().as<Core::Object>();
    }

    using In_tensor = Data::TensorProperty<"Tensor", "Tensor to inspect.">;
    using Out_text = Data::StringProperty<"Text", "Formatted PyTorch tensor.">;

    TensorToTextProcessor()
    {
        RegisterInput<In_tensor>();
        RegisterOutput<Out_text>();
    }

    void Process(Data::Properties const& inputs, Data::Properties outputs)
    {
        outputs.Set<Out_text>(PyTorch::TensorToText(inputs.Get<In_tensor>()));
    }
};
