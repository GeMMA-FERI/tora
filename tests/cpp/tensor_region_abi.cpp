#include <Core/InternalImplementations.hxx>
#include "TensorImpl.hpp"
#include <iostream>

int main()
{
    try {
        auto source = PyTorch::FromTorchTensor(at::arange(12, at::kLong).reshape({3, 4}));
        std::vector<Core::Int64> origin{1, 1}, shape{2, 2};
        Data::Array originArray(origin.data(), origin.size(), sizeof(Core::Int64));
        Data::Array shapeArray(shape.data(), shape.size(), sizeof(Core::Int64));
        auto region = source.ReadRegion(originArray, shapeArray);
        auto values = static_cast<const int64_t*>(region.GetHostData());
        if (values[0] != 5 || values[1] != 6 || values[2] != 9 || values[3] != 10)
            throw std::runtime_error("ABI region values");
        auto original = static_cast<const int64_t*>(source.GetHostData());
        PyTorch::ToTorchTensor(region).fill_(99);
        if (original[5] != 5) throw std::runtime_error("ABI ownership");
        auto expectFailure = [&](Data::Interfaces::IArray const* first, Data::Interfaces::IArray const* second) {
            Data::Tensor result = nullptr;
            Core::ErrorCode error;
            source.get_abi()->_abi_ReadRegion(first, second, &result, &error);
            if (error == nullptr || result != nullptr) throw std::runtime_error("ABI validation");
        };
        expectFailure(nullptr, shapeArray.get_abi());
        Data::Array wrongStride(origin.data(), origin.size(), sizeof(Core::Int32));
        expectFailure(wrongStride.get_abi(), shapeArray.get_abi());
        Data::Array wrongRank(origin.data(), 1, sizeof(Core::Int64));
        expectFailure(wrongRank.get_abi(), shapeArray.get_abi());
        auto view = PyTorch::FromTorchTensor(PyTorch::ToTorchTensor(source).transpose(0, 1));
        void const* pointer = nullptr;
        Core::ErrorCode error;
        view.get_abi()->_abi_GetHostData(&pointer, &error);
        if (error == nullptr || pointer != nullptr) throw std::runtime_error("ABI host layout validation");
        std::cout << "Tensor region ABI checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
