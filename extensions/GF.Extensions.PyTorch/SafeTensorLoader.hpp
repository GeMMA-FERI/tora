#pragma once

#include <ATen/ATen.h>
#include <rapidjson/document.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace PyTorch {
    inline std::filesystem::path ResolveTensorFile(const std::string& file, const std::string& baseDirectory = "")
    {
        if (file.empty()) throw std::runtime_error("Load Tensor: File must not be empty.");
        if (file.find('\0') != std::string::npos || baseDirectory.find('\0') != std::string::npos)
            throw std::runtime_error("Load Tensor: paths must not contain NUL bytes.");
        auto path = std::filesystem::path(std::u8string(file.begin(), file.end()));
        if (!path.is_absolute()) {
            const auto base = std::filesystem::path(std::u8string(baseDirectory.begin(), baseDirectory.end()));
            if (!base.is_absolute() || path.has_root_name() || path.has_root_directory())
                throw std::runtime_error("Load Tensor: a relative File requires an absolute Base Directory.");
            path = base / path;
        }
        return path.lexically_normal();
    }

    namespace safetensor_detail {
        struct Entry {
            std::string name;
            at::ScalarType type;
            std::vector<int64_t> shape;
            uint64_t begin, end;
        };

        inline at::ScalarType Dtype(const std::string& name)
        {
            if (name == "BOOL") return at::kBool;
            if (name == "U8") return at::kByte;
            if (name == "I8") return at::kChar;
            if (name == "I16") return at::kShort;
            if (name == "I32") return at::kInt;
            if (name == "I64") return at::kLong;
            if (name == "U16") return at::ScalarType::UInt16;
            if (name == "U32") return at::ScalarType::UInt32;
            if (name == "U64") return at::ScalarType::UInt64;
            if (name == "F16") return at::kHalf;
            if (name == "BF16") return at::kBFloat16;
            if (name == "F32") return at::kFloat;
            if (name == "F64") return at::kDouble;
            throw std::runtime_error("Unsupported Safetensors dtype: " + name);
        }

        inline void Require(bool condition, const char* message)
        {
            if (!condition) throw std::runtime_error(message);
        }

        inline void Read(std::ifstream& stream, void* destination, uint64_t bytes)
        {
            auto* cursor = static_cast<char*>(destination);
            while (bytes) {
                const auto count = static_cast<std::streamsize>(std::min<uint64_t>(bytes, 64 * 1024 * 1024));
                stream.read(cursor, count);
                Require(stream.gcount() == count, "Truncated Safetensors file.");
                cursor += count;
                bytes -= count;
            }
        }

        inline void UniqueKeys(const rapidjson::Value& object)
        {
            std::set<std::string> names;
            for (const auto& member : object.GetObject())
                Require(names.emplace(member.name.GetString(), member.name.GetStringLength()).second,
                    "Duplicate key in Safetensors header.");
        }
    }

    // Reads the selected byte range into owned, contiguous CPU storage. The
    // complete header is validated; no other tensor payload is allocated/read.
    inline at::Tensor LoadSafeTensor(const std::filesystem::path& path, const std::string& tensorName)
    {
        using namespace safetensor_detail;
        static_assert(std::endian::native == std::endian::little);
        try {
            Require(!tensorName.empty(), "Tensor Name must not be empty.");
            std::ifstream stream(path, std::ios::binary | std::ios::ate);
            Require(stream.is_open(), "Cannot open Safetensors file.");
            const auto length = stream.tellg();
            Require(length >= std::streampos(8), "Invalid Safetensors file length.");
            const auto fileSize = static_cast<uint64_t>(length);
            stream.seekg(0);
            uint64_t headerSize = 0;
            Read(stream, &headerSize, 8);
            Require(headerSize > 0 && headerSize <= 100'000'000 && headerSize <= fileSize - 8,
                "Invalid Safetensors header length (maximum 100 MB).");
            std::string header(static_cast<size_t>(headerSize), '\0');
            Read(stream, header.data(), headerSize);
            Require(header.front() == '{', "Safetensors header must start with an object.");
            Require(header.find('\0') == std::string::npos, "NUL byte in Safetensors header.");
            rapidjson::Document document;
            document.Parse<rapidjson::kParseValidateEncodingFlag>(header.data(), header.size());
            Require(!document.HasParseError() && document.IsObject(), "Invalid Safetensors JSON header.");
            UniqueKeys(document);
            const uint64_t dataSize = fileSize - 8 - headerSize;
            std::vector<Entry> entries;
            for (const auto& member : document.GetObject()) {
                const std::string name(member.name.GetString(), member.name.GetStringLength());
                const auto& value = member.value;
                Require(value.IsObject(), "Invalid Safetensors entry.");
                UniqueKeys(value);
                if (name == "__metadata__") {
                    for (const auto& item : value.GetObject()) Require(item.value.IsString(), "Metadata values must be strings.");
                    continue;
                }
                Require(value.HasMember("dtype") && value["dtype"].IsString()
                    && value.HasMember("shape") && value["shape"].IsArray()
                    && value.HasMember("data_offsets") && value["data_offsets"].IsArray()
                    && value["data_offsets"].Size() == 2, "Invalid tensor descriptor.");
                Entry entry{ name, Dtype(std::string(value["dtype"].GetString(), value["dtype"].GetStringLength())), {}, 0, 0 };
                bool empty = false;
                for (const auto& dimension : value["shape"].GetArray()) {
                    Require(dimension.IsUint64() && dimension.GetUint64() <= INT64_MAX, "Invalid tensor dimension.");
                    entry.shape.push_back(static_cast<int64_t>(dimension.GetUint64()));
                    empty |= dimension.GetUint64() == 0;
                }
                uint64_t elements = empty ? 0 : 1;
                for (auto dimension : entry.shape) {
                    Require(dimension == 0 || elements <= INT64_MAX / static_cast<uint64_t>(dimension), "Tensor shape overflows.");
                    elements *= static_cast<uint64_t>(dimension);
                }
                const auto width = c10::elementSize(entry.type);
                Require(elements <= INT64_MAX / width, "Tensor byte count overflows.");
                const auto& offsets = value["data_offsets"];
                Require(offsets[0].IsUint64() && offsets[1].IsUint64(), "Invalid tensor offsets.");
                entry.begin = offsets[0].GetUint64();
                entry.end = offsets[1].GetUint64();
                Require(entry.begin <= entry.end && entry.end <= dataSize && entry.end - entry.begin == elements * width,
                    "Tensor offsets do not match its shape, dtype or file size.");
                entries.push_back(std::move(entry));
            }
            std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
                return a.begin == b.begin ? a.end < b.end : a.begin < b.begin;
            });
            uint64_t end = 0;
            for (const auto& entry : entries) {
                Require(entry.begin == end, "Overlapping tensors or gaps in Safetensors data.");
                end = entry.end;
            }
            Require(end == dataSize, "Unreferenced bytes in Safetensors file.");
            const auto selected = std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) { return entry.name == tensorName; });
            Require(selected != entries.end(), "Tensor Name was not found in the file.");
            auto tensor = at::empty(selected->shape, at::TensorOptions().dtype(selected->type).device(at::kCPU));
            stream.seekg(static_cast<std::streamoff>(8 + headerSize + selected->begin));
            Require(stream.good(), "Cannot seek to tensor data.");
            Read(stream, tensor.data_ptr(), selected->end - selected->begin);
            if (selected->type == at::kBool) {
                const auto* bytes = static_cast<const unsigned char*>(tensor.const_data_ptr());
                for (uint64_t i = 0; i < selected->end - selected->begin; ++i)
                    Require(bytes[i] <= 1, "Invalid BOOL tensor byte (expected 0 or 1).");
            }
            return tensor;
        } catch (const std::exception& error) {
            const auto utf8 = path.u8string();
            throw std::runtime_error("Load Tensor [" + std::string(utf8.begin(), utf8.end()) + ", " + tensorName + "]: " + error.what());
        }
    }
}
