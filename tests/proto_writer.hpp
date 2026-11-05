#pragma once

// Tiny protobuf encoder for tests: lets a test build exact byte sequences (edge cases the
// python onnx package would never write) without depending on libprotobuf.

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace test_proto {

class Writer {
public:
    Writer& varint(std::uint64_t v) {
        while (v >= 0x80) {
            buf_.push_back(static_cast<char>((v & 0x7F) | 0x80));
            v >>= 7;
        }
        buf_.push_back(static_cast<char>(v));
        return *this;
    }
    Writer& tag(std::uint32_t field, std::uint8_t wire) {
        return varint((static_cast<std::uint64_t>(field) << 3) | wire);
    }
    Writer& int_field(std::uint32_t field, std::int64_t v) {
        return tag(field, 0).varint(static_cast<std::uint64_t>(v));
    }
    Writer& bytes_field(std::uint32_t field, std::string_view v) {
        tag(field, 2).varint(v.size());
        buf_.append(v);
        return *this;
    }
    Writer& message(std::uint32_t field, const Writer& m) { return bytes_field(field, m.str()); }
    Writer& float_field(std::uint32_t field, float f) {
        tag(field, 5);
        char b[4];
        std::memcpy(b, &f, 4);
        buf_.append(b, 4);
        return *this;
    }
    Writer& raw(std::string_view bytes) {
        buf_.append(bytes);
        return *this;
    }

    [[nodiscard]] const std::string& str() const { return buf_; }
    [[nodiscard]] const std::uint8_t* data() const {
        return reinterpret_cast<const std::uint8_t*>(buf_.data());
    }
    [[nodiscard]] std::size_t size() const { return buf_.size(); }

private:
    std::string buf_;
};

}  // namespace test_proto
