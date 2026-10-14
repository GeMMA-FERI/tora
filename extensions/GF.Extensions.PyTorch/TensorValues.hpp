#pragma once
#include <ATen/ATen.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace PyTorch {
    inline std::vector<std::string> ReadTensorValues(const at::Tensor& tensor, uint64_t start, uint64_t count, bool allowDeviceCopy)
    {
        if (!tensor.defined() || tensor.layout() != at::kStrided || tensor.is_quantized() || tensor.is_meta())
            throw std::runtime_error("Element previews require a dense, non-quantized tensor with storage.");
        const auto size = static_cast<uint64_t>(tensor.numel());
        if (count > 256 || start > size || count > size - start)
            throw std::runtime_error("Invalid tensor preview range (maximum 256 values).");
        if (!tensor.device().is_cpu() && !allowDeviceCopy)
            throw std::runtime_error("Reading device tensors requires explicit permission to copy the preview to CPU.");
        std::vector<std::string> result;
        if (count == 0) return result;
        // Select logical elements through views; flattening a non-contiguous
        // tensor first could copy the entire allocation. Only stack the page.
        std::vector<at::Tensor> selected;
        auto source = tensor.detach();
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t index = start + i;
            std::vector<int64_t> coordinates(source.dim());
            for (int64_t dim = source.dim(); dim-- > 0;) {
                coordinates[dim] = index % source.size(dim);
                index /= source.size(dim);
            }
            auto value = source;
            for (auto coordinate : coordinates) value = value.select(0, coordinate);
            selected.push_back(value);
        }
        auto page = at::stack(selected).to(at::kCPU);
        for (int64_t i = 0; i < page.numel(); ++i) {
            std::ostringstream text;
            text << std::setprecision(std::numeric_limits<double>::max_digits10);
            switch (page.scalar_type()) {
                case at::kBool: text << (page[i].item<bool>() ? "true" : "false"); break;
                case at::kByte:
                case at::ScalarType::UInt16:
                case at::ScalarType::UInt32: text << page[i].item<int64_t>(); break;
                case at::ScalarType::UInt64: text << static_cast<const uint64_t*>(page.const_data_ptr())[i]; break;
                case at::kChar:
                case at::kShort:
                case at::kInt:
                case at::kLong: text << page[i].item<int64_t>(); break;
                case at::kHalf:
                case at::kBFloat16:
                case at::kFloat:
                case at::kDouble: text << page[i].item<double>(); break;
                case at::kComplexHalf:
                case at::kComplexFloat:
                case at::kComplexDouble: text << page[i].item<c10::complex<double>>(); break;
                default: throw std::runtime_error("Unsupported tensor dtype for element preview.");
            }
            result.push_back(text.str());
        }
        return result;
    }
}
