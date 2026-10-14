#include "TensorRegion.hpp"
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
        auto region = PyTorch::ReadTensorRegion(tensor, {2, 3}, {4, 5});
        Check(region.sizes() == at::IntArrayRef({4, 5}) && region.is_contiguous(), "Region shape and layout");
        Check(region[0][0].item<int64_t>() == 63 && region[3][4].item<int64_t>() == 157, "Region coordinates");
        region.fill_(-1);
        Check(tensor[2][3].item<int64_t>() == 63, "Independent snapshot");
        auto view = tensor.transpose(0, 1).slice(0, 2, 4);
        region = PyTorch::ReadTensorRegion(view, {0, 18}, {2, 2});
        Check(at::equal(region, at::tensor({542, 572, 543, 573}, at::kLong).reshape({2, 2})), "Strides and storage offset");
        auto scalar = at::scalar_tensor(1.25, at::kDouble);
        region = PyTorch::ReadTensorRegion(scalar, {}, {});
        Check(region.dim() == 0 && region.item<double>() == 1.25, "Scalar region");
        region.fill_(0);
        Check(scalar.item<double>() == 1.25, "Scalar snapshot");
        Check(PyTorch::ReadTensorRegion(tensor, {20, 30}, {0, 0}).numel() == 0, "Empty boundary region");
        Check(PyTorch::ReadTensorRegion(at::empty({0, 3}), {0, 0}, {0, 3}).numel() == 0, "Empty source");
        for (auto type : {at::kBool, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong, at::kHalf,
            at::kBFloat16, at::kFloat, at::kDouble, at::kComplexFloat, at::kComplexDouble}) {
            auto source = at::ones({2, 3}, type);
            auto copy = PyTorch::ReadTensorRegion(source, {0, 1}, {2, 2});
            Check(copy.scalar_type() == type && at::equal(copy, source.slice(1, 1, 3)), "Dtype preservation");
        }
        auto unsignedValue = at::empty({1}, at::ScalarType::UInt64);
        static_cast<uint64_t*>(unsignedValue.mutable_data_ptr())[0] = UINT64_MAX;
        region = PyTorch::ReadTensorRegion(unsignedValue, {0}, {1});
        Check(*static_cast<const uint64_t*>(region.const_data_ptr()) == UINT64_MAX, "Unsigned precision");
        auto conjugate = at::scalar_tensor(c10::complex<double>(1.5, -2), at::kComplexDouble).conj();
        region = PyTorch::ReadTensorRegion(conjugate, {}, {});
        Check(!region.is_conj() && region.item<c10::complex<double>>() == c10::complex<double>(1.5, 2), "Resolve conjugation");
        Reject([&] { PyTorch::ReadTensorRegion(tensor, {0}, {1}); });
        Reject([&] { PyTorch::ReadTensorRegion(tensor, {-1, 0}, {1, 1}); });
        Reject([&] { PyTorch::ReadTensorRegion(tensor, {0, 0}, {-1, 1}); });
        Reject([&] { PyTorch::ReadTensorRegion(tensor, {19, 0}, {2, 1}); });
        Reject([&] { PyTorch::ReadTensorRegion(tensor, {INT64_MAX, 0}, {INT64_MAX, 1}); });
        Reject([&] { PyTorch::ReadTensorRegion(at::Tensor(), {}, {}); });
        Reject([&] { PyTorch::ReadTensorRegion(at::empty({2}, at::TensorOptions().device(at::kMeta)), {0}, {1}); });
        Reject([&] { PyTorch::ReadTensorRegion(tensor.to_sparse(), {0, 0}, {1, 1}); });
        Reject([&] { PyTorch::ReadTensorRegion(at::quantize_per_tensor(at::ones({2}), 0.1, 0, at::kQUInt8), {0}, {1}); });
        std::cout << "Tensor region checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
