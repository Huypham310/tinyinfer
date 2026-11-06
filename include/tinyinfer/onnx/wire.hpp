#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tinyinfer::onnx {

/// Thrown for malformed or unsupported model files.
class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Protobuf wire types (groups, 3 and 4, are deprecated and rejected).
enum class WireType : std::uint8_t { Varint = 0, Fixed64 = 1, Bytes = 2, Fixed32 = 5 };

/// Minimal protobuf wire-format decoder over a byte range it does not own.
///
/// We only need to *read* a handful of ONNX messages, so instead of pulling in libprotobuf
/// (and protoc in the build) this walks the encoding directly: every field is a varint tag
/// (field_number << 3 | wire_type) followed by a payload whose size the wire type tells us.
/// Length-delimited payloads are returned as views into the original buffer (no copies), and
/// every read is bounds-checked so a truncated file throws instead of reading past the end.
class WireReader {
public:
    struct Field {
        std::uint32_t number;
        WireType type;
    };

    WireReader(const std::uint8_t* data, std::size_t size) noexcept
        : pos_(data), end_(data + size), begin_(data) {}
    explicit WireReader(std::string_view bytes) noexcept
        : WireReader(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()) {}

    [[nodiscard]] bool done() const noexcept { return pos_ >= end_; }
    [[nodiscard]] std::size_t offset() const noexcept {
        return static_cast<std::size_t>(pos_ - begin_);
    }

    /// Reads the next tag. Call only when `!done()`.
    Field next();

    std::uint64_t varint();
    /// int32/int64/enum fields are plain varints; negatives take all 10 bytes (two's complement).
    std::int64_t int64() { return static_cast<std::int64_t>(varint()); }
    std::uint32_t fixed32();
    std::uint64_t fixed64();
    float f32();
    double f64();
    /// Payload of a length-delimited field, as a view into the buffer.
    std::string_view bytes();
    /// Same payload, as a reader for an embedded message.
    WireReader message() { return WireReader(bytes()); }

    /// Skips the payload of a field we don't care about.
    void skip(WireType type);

private:
    void need(std::size_t n) const;

    const std::uint8_t* pos_;
    const std::uint8_t* end_;
    const std::uint8_t* begin_;
};

}  // namespace tinyinfer::onnx
