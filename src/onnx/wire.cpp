#include "tinyinfer/onnx/wire.hpp"

#include <cstring>

namespace tinyinfer::onnx {

void WireReader::need(std::size_t n) const {
    if (static_cast<std::size_t>(end_ - pos_) < n) {
        throw ParseError("truncated protobuf: need " + std::to_string(n) + " bytes at offset " +
                         std::to_string(offset()) + ", have " +
                         std::to_string(end_ - pos_));
    }
}

std::uint64_t WireReader::varint() {
    std::uint64_t value = 0;
    // 7 payload bits per byte, high bit = "more bytes follow"; 10 bytes cover 64 bits
    for (int shift = 0; shift < 64; shift += 7) {
        need(1);
        const std::uint8_t b = *pos_++;
        value |= static_cast<std::uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) return value;
    }
    throw ParseError("varint longer than 10 bytes at offset " + std::to_string(offset()));
}

WireReader::Field WireReader::next() {
    const std::uint64_t tag = varint();
    const auto type = static_cast<std::uint8_t>(tag & 0x7);
    const auto number = static_cast<std::uint32_t>(tag >> 3);
    if (number == 0) throw ParseError("field number 0 at offset " + std::to_string(offset()));
    switch (type) {
        case 0:
        case 1:
        case 2:
        case 5:
            return {number, static_cast<WireType>(type)};
        default:
            throw ParseError("unsupported wire type " + std::to_string(type) + " for field " +
                             std::to_string(number));
    }
}

std::uint32_t WireReader::fixed32() {
    need(4);
    // protobuf is little-endian on the wire; so is every target we build for
    std::uint32_t v;
    std::memcpy(&v, pos_, 4);
    pos_ += 4;
    return v;
}

std::uint64_t WireReader::fixed64() {
    need(8);
    std::uint64_t v;
    std::memcpy(&v, pos_, 8);
    pos_ += 8;
    return v;
}

float WireReader::f32() {
    const std::uint32_t bits = fixed32();
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

double WireReader::f64() {
    const std::uint64_t bits = fixed64();
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
}

std::string_view WireReader::bytes() {
    const std::uint64_t len = varint();
    need(static_cast<std::size_t>(len));
    std::string_view out(reinterpret_cast<const char*>(pos_), static_cast<std::size_t>(len));
    pos_ += len;
    return out;
}

void WireReader::skip(WireType type) {
    switch (type) {
        case WireType::Varint:
            varint();
            break;
        case WireType::Fixed64:
            need(8);
            pos_ += 8;
            break;
        case WireType::Bytes:
            bytes();
            break;
        case WireType::Fixed32:
            need(4);
            pos_ += 4;
            break;
    }
}

}  // namespace tinyinfer::onnx
