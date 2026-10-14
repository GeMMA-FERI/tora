#pragma once
#include <ATen/ATen.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace PyTorch
{
    inline at::Tensor ReadTensorRegion(const at::Tensor& tensor,
        const std::vector<int64_t>& origin, const std::vector<int64_t>& shape)
    {
        if (!tensor.defined() || tensor.layout() != at::kStrided || tensor.is_quantized() || tensor.is_meta())
            throw std::runtime_error("Region reads require a dense, non-quantized tensor with storage.");
        if (origin.size() != tensor.dim() || shape.size() != tensor.dim())
            throw std::runtime_error("Region origin and shape must match the tensor rank.");
        bool empty = false;
        for (size_t dim = 0; dim < shape.size(); ++dim) {
            if (origin[dim] < 0 || shape[dim] < 0 || origin[dim] > tensor.size(dim) || shape[dim] > tensor.size(dim) - origin[dim])
                throw std::runtime_error("Tensor region is out of bounds.");
            empty |= shape[dim] == 0;
        }
        size_t bytes = tensor.element_size();
        if (!empty) {
            for (auto size : shape) {
                if (static_cast<uint64_t>(size) > std::numeric_limits<size_t>::max() / bytes)
                    throw std::overflow_error("Tensor region byte size overflow.");
                bytes *= static_cast<size_t>(size);
            }
        }
        auto region = tensor.detach();
        for (size_t dim = 0; dim < shape.size(); ++dim)
            region = region.narrow(dim, origin[dim], shape[dim]);
        // copy=true guarantees independent storage, including an already-contiguous CPU region.
        return region.resolve_conj().resolve_neg().to(at::kCPU, region.scalar_type(), false, true, at::MemoryFormat::Contiguous);
    }
}
