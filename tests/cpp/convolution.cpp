#include "pch.h"
#include "Processors.hpp"
#include <iostream>

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename T>
auto Processor() {
    return T::create_instance().template as<GF::Api::Extensions::Processing::DataProcessor>();
}

int main() {
    try {
        auto one = Processor<Conv1d>();
        auto ins = one.CreateInputs();
        auto outs = one.CreateOutputs();
        auto x = at::arange(16, at::kFloat).reshape({1, 2, 8});
        auto weight = at::ones({2, 1, 3});
        ins.Set<Conv1d::In_input>(PyTorch::FromTorchTensor(x));
        ins.Set<Conv1d::In_weight>(PyTorch::FromTorchTensor(weight));
        Check(ins.Get<Conv1d::In_stride_0>() == 1 && ins.Get<Conv1d::In_padding_0>() == 0
            && ins.Get<Conv1d::In_dilation_0>() == 1 && ins.Get<Conv1d::In_groups>() == 1, "Conv1d defaults");
        ins.Set<Conv1d::In_groups>(Core::Int64(2));
        ins.Set<Conv1d::In_stride_0>(Core::Int64(2));
        ins.Set<Conv1d::In_padding_0>(Core::Int64(1));
        one.Process(ins, outs);
        Check(at::equal(PyTorch::ToTorchTensor(outs.Get<Conv1d::Out_output_0>()), at::conv1d(x, weight, {}, {2}, {1}, {1}, 2)), "Conv1d output");

        auto two = Processor<Conv2d>();
        ins = two.CreateInputs(); outs = two.CreateOutputs();
        x = at::arange(72, at::kFloat).reshape({1, 1, 8, 9});
        weight = at::arange(6, at::kFloat).reshape({1, 1, 3, 2});
        ins.Set<Conv2d::In_input>(PyTorch::FromTorchTensor(x));
        ins.Set<Conv2d::In_weight>(PyTorch::FromTorchTensor(weight));
        Check(ins.Get<Conv2d::In_stride_0>() == 1 && ins.Get<Conv2d::In_stride_1>() == 1
            && ins.Get<Conv2d::In_padding_0>() == 0 && ins.Get<Conv2d::In_padding_1>() == 0
            && ins.Get<Conv2d::In_dilation_0>() == 1 && ins.Get<Conv2d::In_dilation_1>() == 1, "Conv2d defaults");
        two.Process(ins, outs);
        Check(at::equal(PyTorch::ToTorchTensor(outs.Get<Conv2d::Out_output_0>()), at::conv2d(x, weight)), "Conv2d default output");
        ins.Set<Conv2d::In_stride_0>(Core::Int64(2));
        ins.Set<Conv2d::In_padding_0>(Core::Int64(1));
        ins.Set<Conv2d::In_dilation_1>(Core::Int64(2));
        auto bias = at::full({1}, 0.5, at::kFloat);
        ins.Set<Conv2d::In_bias>(PyTorch::FromTorchTensor(bias));
        two.Process(ins, outs);
        Check(at::equal(PyTorch::ToTorchTensor(outs.Get<Conv2d::Out_output_0>()), at::conv2d(x, weight, bias, {2, 1}, {1, 0}, {1, 2})), "Conv2d axis order and bias");
        std::cout << "Generated Conv1d/Conv2d processors: defaults and outputs verified.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
