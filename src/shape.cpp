#include "tinyinfer/shape.hpp"

#include <stdexcept>

namespace tinyinfer {

Shape::Shape(std::initializer_list<std::int64_t> dims) : Shape(std::vector<std::int64_t>(dims)) {}

Shape::Shape(std::vector<std::int64_t> dims) : dims_(std::move(dims)) {
    for (auto d : dims_) {
        // ONNX uses -1 / 0 as placeholders in reshape specs, but a real tensor never has them
        if (d < 0) throw std::invalid_argument("negative dimension in shape " + to_string());
    }
}

std::int64_t Shape::numel() const noexcept {
    std::int64_t n = 1;
    for (auto d : dims_) n *= d;
    return n;
}

std::vector<std::int64_t> Shape::contiguous_strides() const {
    std::vector<std::int64_t> strides(dims_.size(), 1);
    for (std::size_t i = dims_.size(); i-- > 1;) strides[i - 1] = strides[i] * dims_[i];
    return strides;
}

std::string Shape::to_string() const {
    std::string s = "[";
    for (std::size_t i = 0; i < dims_.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(dims_[i]);
    }
    return s + "]";
}

}  // namespace tinyinfer
