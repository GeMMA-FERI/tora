#include "pch.h"
#include "Processors.hpp"
#include "LoadTensorProcessor.hpp"
#include <iostream>
#include <format>
#include <map>

template<typename T>
auto Processor()
{
    return T::create_instance().template as<GF::Api::Extensions::Processing::DataProcessor>();
}

template<typename T>
bool Matches(const std::string& id)
{
    const auto uuid = T::implementation_id();
    const auto compiled = std::format("{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid.Data1, uuid.Data2, uuid.Data3, uuid.Data4[0], uuid.Data4[1], uuid.Data4[2], uuid.Data4[3],
        uuid.Data4[4], uuid.Data4[5], uuid.Data4[6], uuid.Data4[7]);
    return compiled == id;
}

// A small test harness for this exported graph, outside the Node Editor.
// It passes serialized properties and edge keys through the real SDK processors.
int main()
{
    try {
        const auto directory = std::filesystem::path("_build/captured-module-test");
        std::ifstream file(directory / "model.nodegraph.json");
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        rapidjson::Document graph;
        graph.Parse(text.c_str());
        if (graph.HasParseError() || !graph.IsObject()) throw std::runtime_error("Invalid captured graph.");
        const auto input = PyTorch::LoadSafeTensor(directory / "reference.safetensors", "input");
        const auto expected = PyTorch::LoadSafeTensor(directory / "reference.safetensors", "expected");
        std::map<std::pair<std::string, std::string>, Data::Variant> values;
        int checked = 0, loaded = 0;
        for (const auto& member : graph["Nodes"].GetObject()) {
            const std::string node = member.name.GetString();
            const auto& definition = member.value;
            const std::string implementation = definition["NodeBase"].GetString();
            if (implementation == "86e1a1e0-2ad2-4096-a707-0ed6302b9bac") {
                values[{node, "x"}] = Data::Variant(PyTorch::FromTorchTensor(input).get_abi());
                continue;
            }
            if (implementation == "e036c9af-ae18-4b4c-bd6d-c4b77691ba33") {
                for (const auto& edge : graph["Edges"].GetObject()) {
                    const auto& e = edge.value;
                    if (node != e["V"].GetString()) continue;
                    const auto actual = PyTorch::ToTorchTensor(values.at({e["U"].GetString(), e["UT"].GetString()}).as<Data::Tensor>());
                    if (!at::allclose(actual, expected, 1e-5, 1e-6)) throw std::runtime_error("Captured graph differs from Python.");
                    ++checked;
                }
                continue;
            }
            decltype(Processor<Conv2d>()) processor = nullptr;
            if (Matches<LoadTensorProcessor>(implementation)) { processor = Processor<LoadTensorProcessor>(); ++loaded; }
            else if (Matches<Conv2d>(implementation)) processor = Processor<Conv2d>();
            else if (Matches<Add>(implementation)) processor = Processor<Add>();
            else if (Matches<Relu>(implementation)) processor = Processor<Relu>();
            else throw std::runtime_error("Unexpected processor in captured graph.");
            auto inputs = processor.CreateInputs();
            auto outputs = processor.CreateOutputs();
            for (const auto& property : definition["Properties"].GetObject()) {
                const std::string key = property.name.GetString();
                const auto& value = property.value;
                if (value.IsString()) inputs.SetPropertyValue(key, std::string(value.GetString()));
                else if (Data::details::IsIntegral(inputs.GetPropertyValueType(key))) inputs.SetPropertyValue(key, Core::Int64(value.GetInt64()));
                else if (Data::details::IsFloatingPoint(inputs.GetPropertyValueType(key))) inputs.SetPropertyValue(key, value.GetDouble());
                else throw std::runtime_error("Unexpected property type in captured graph.");
            }
            for (const auto& edge : graph["Edges"].GetObject()) {
                const auto& e = edge.value;
                if (node == e["V"].GetString()) inputs.SetPropertyValue(e["VT"].GetString(), values.at({e["U"].GetString(), e["UT"].GetString()}));
            }
            processor.Process(inputs, outputs);
            for (const auto& key : outputs.GetPropertyKeys()) values[{node, key}] = outputs.GetPropertyValue(key);
        }
        if (checked != 1 || loaded != 3) throw std::runtime_error("Missing model output or tensor resources.");
        std::cout << "Captured graph: three resource loaders and generated Conv2d/Add/ReLU match Python.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
