#pragma once

#include "Helpers.hpp"

struct TensorFromTypedArrayProcessor : GF::Extensions::impl::DataProcessorBridgeImpl<TensorFromTypedArrayProcessor> {
    static constexpr auto name() { return std::string_view("Tensor From Typed Array"); }
    static constexpr auto description() { return std::string_view("Creates a CPU PyTorch tensor from flat primitive values and integral dimensions."); }
    static constexpr auto interface_id() { return Core::interfaces::interface_id_storage<GF::Interfaces::Ext::IDataProcessor>::value; }

    // {CABBB062-8A93-44C3-A00D-BC1CC438FCBB}
    static constexpr auto implementation_id() {
        return Core::Uuid(0xcabbb062, 0x8a93, 0x44c3, { 0xa0, 0x0d, 0xbc, 0x1c, 0xc4, 0x38, 0xfc, 0xbb });
    }

    static auto create_instance() { return Core::make_base_obj<TensorFromTypedArrayProcessor>().as<Core::Object>(); }

    using In_values = Data::TypedArrayProperty<"Values", "Flat array of primitive numeric values. The array element type becomes the tensor dtype.">;
    using In_dimensions = Data::TypedArrayProperty<"Dimensions", "Integral tensor dimensions. Their product must equal the number of values.">;
    using Out_tensor = Data::TensorProperty<"Tensor", "CPU tensor containing an owned copy of the input values.">;

    TensorFromTypedArrayProcessor()
    {
        RegisterInput<In_values>();
        RegisterInput<In_dimensions>();
        RegisterOutput<Out_tensor>();
    }

    void Process(Data::Properties const& inputs, Data::Properties outputs)
    {
        outputs.Set<Out_tensor>(PyTorch::TensorFromTypedArray(inputs.Get<In_values>(), inputs.Get<In_dimensions>()));
    }
};
