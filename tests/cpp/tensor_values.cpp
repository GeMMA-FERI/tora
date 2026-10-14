#include "TensorValues.hpp"
#include <iostream>

static void Check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<typename Function>
static void Reject(Function function)
{
    bool rejected = false;
    try { function(); }
    catch (const std::exception&) { rejected = true; }
    Check(rejected, "Expected an unsupported tensor or invalid range to fail.");
}

int main()
{
    try {
        auto tensor = at::arange(600, at::kLong).reshape({20, 30});
        auto page = PyTorch::ReadTensorValues(tensor, 256, 256, false);
        Check(page.size() == 256 && page.front() == "256" && page.back() == "511", "Page boundaries");
        auto view = tensor.transpose(0, 1).slice(0, 2, 4);
        page = PyTorch::ReadTensorValues(view, 18, 4, false);
        Check(page == std::vector<std::string>({"542", "572", "3", "33"}), "Strides and storage offset");
        Check(PyTorch::ReadTensorValues(at::scalar_tensor(1.25, at::kDouble), 0, 1, false)[0] == "1.25", "Scalar");
        Check(PyTorch::ReadTensorValues(at::empty({0}), 0, 0, false).empty(), "Empty tensor");
        Check(PyTorch::ReadTensorValues(at::ones({2}, at::kBool), 0, 2, false)[0] == "true", "Boolean");
        auto maximum = std::numeric_limits<int64_t>::max();
        Check(PyTorch::ReadTensorValues(at::scalar_tensor(maximum, at::kLong), 0, 1, false)[0] == std::to_string(maximum), "Integer precision");
        auto unsignedValue = at::empty({1}, at::ScalarType::UInt64);
        unsignedValue.data_ptr<uint64_t>()[0] = std::numeric_limits<uint64_t>::max();
        Check(PyTorch::ReadTensorValues(unsignedValue, 0, 1, false)[0] == "18446744073709551615", "Unsigned precision");
        auto complex = at::scalar_tensor(c10::complex<double>(1.5, -2.0), at::kComplexDouble);
        Check(!PyTorch::ReadTensorValues(complex, 0, 1, false)[0].empty(), "Complex");
        Check(PyTorch::ReadTensorValues(tensor, 590, 10, false).back() == "599", "Last page");
        Reject([&] { PyTorch::ReadTensorValues(tensor, 0, 257, false); });
        Reject([&] { PyTorch::ReadTensorValues(tensor, 599, 2, false); });
        Reject([&] { PyTorch::ReadTensorValues(tensor, UINT64_MAX, 1, false); });
        Reject([&] { PyTorch::ReadTensorValues(at::Tensor(), 0, 0, false); });
        Reject([&] { PyTorch::ReadTensorValues(at::empty({2}, at::TensorOptions().device(at::kMeta)), 0, 1, false); });
        Reject([&] { PyTorch::ReadTensorValues(tensor.to_sparse(), 0, 1, false); });
        std::cout << "Tensor value preview checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
