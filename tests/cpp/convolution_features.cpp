#include "pch.h"
#include "Processors.hpp"
#include "SafeTensorLoader.hpp"
#include <iostream>

template<typename T>
auto Processor()
{
    return T::create_instance().template as<GF::Api::Extensions::Processing::DataProcessor>();
}

int main()
{
    try {
        std::ifstream file("examples/conv2d-features.json");
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        rapidjson::Document config;
        config.Parse(text.c_str());
        if (config.HasParseError() || !config.IsObject()) throw std::runtime_error("Cannot read example configuration.");
        const auto load = [](const std::string& key) {
            return PyTorch::LoadSafeTensor(std::filesystem::path("_build/conv2d-features.safetensors"), key);
        };
        auto current = load("input");
        for (const auto& stage : config["stages"].GetArray()) {
            const std::string name = stage["name"].GetString();
            auto conv = Processor<Conv2d>();
            auto inputs = conv.CreateInputs();
            auto outputs = conv.CreateOutputs();
            inputs.Set<Conv2d::In_input>(PyTorch::FromTorchTensor(current));
            inputs.Set<Conv2d::In_weight>(PyTorch::FromTorchTensor(load(name + ".weight")));
            inputs.Set<Conv2d::In_bias>(PyTorch::FromTorchTensor(load(name + ".bias")));
            inputs.Set<Conv2d::In_stride_0>(Core::Int64(stage["stride"][0].GetInt64()));
            inputs.Set<Conv2d::In_stride_1>(Core::Int64(stage["stride"][1].GetInt64()));
            inputs.Set<Conv2d::In_padding_0>(Core::Int64(stage["padding"][0].GetInt64()));
            inputs.Set<Conv2d::In_padding_1>(Core::Int64(stage["padding"][1].GetInt64()));
            inputs.Set<Conv2d::In_dilation_0>(Core::Int64(stage["dilation"][0].GetInt64()));
            inputs.Set<Conv2d::In_dilation_1>(Core::Int64(stage["dilation"][1].GetInt64()));
            inputs.Set<Conv2d::In_groups>(Core::Int64(stage["groups"].GetInt64()));
            conv.Process(inputs, outputs);
            current = PyTorch::ToTorchTensor(outputs.Get<Conv2d::Out_output_0>());
            auto expected = PyTorch::LoadSafeTensor(std::filesystem::path("_build/conv2d-features-reference.safetensors"), name + ".conv");
            if (!at::allclose(current, expected, 1e-4, 1e-6)) throw std::runtime_error(name + " convolution differs from Python.");
            auto relu = Processor<Relu>();
            auto reluInputs = relu.CreateInputs();
            auto reluOutputs = relu.CreateOutputs();
            reluInputs.Set<Relu::In_self>(PyTorch::FromTorchTensor(current));
            relu.Process(reluInputs, reluOutputs);
            current = PyTorch::ToTorchTensor(reluOutputs.Get<Relu::Out_output_0>());
            expected = PyTorch::LoadSafeTensor(std::filesystem::path("_build/conv2d-features-reference.safetensors"), name + ".relu");
            if (!at::allclose(current, expected, 1e-4, 1e-6)) throw std::runtime_error(name + " activation differs from Python.");
            std::cout << name << ": convolution and activation match Python.\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
