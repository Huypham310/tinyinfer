#include "tinyinfer/tensor.hpp"

#include <cstring>

namespace tinyinfer {

std::size_t dtype_size(DType t) noexcept {
    switch (t) {
        case DType::Float32:
        case DType::Int32:
            return 4;
        case DType::Int8:
            return 1;
        case DType::Int64:
            return 8;
    }
    return 0;
}

const char* dtype_name(DType t) noexcept {
    switch (t) {
        case DType::Float32:
            return "float32";
        case DType::Int8:
            return "int8";
        case DType::Int32:
            return "int32";
        case DType::Int64:
            return "int64";
    }
    return "?";
}

Tensor::Tensor(Shape shape, DType dtype)
    : shape_(std::move(shape)),
      dtype_(dtype),
      storage_(std::make_shared<AlignedBuffer>(static_cast<std::size_t>(shape_.numel()) *
                                               dtype_size(dtype))) {}

void Tensor::check_type(DType wanted) const {
    if (wanted != dtype_) {
        throw std::invalid_argument(std::string("tensor is ") + dtype_name(dtype_) +
                                    ", accessed as " + dtype_name(wanted));
    }
}

std::int64_t Tensor::offset_of(const std::vector<std::int64_t>& index) const {
    if (index.size() != shape_.rank()) throw std::out_of_range("index rank != tensor rank");
    const auto strides = shape_.contiguous_strides();
    std::int64_t off = 0;
    for (std::size_t i = 0; i < index.size(); ++i) {
        if (index[i] < 0 || index[i] >= shape_[i]) throw std::out_of_range("index out of bounds");
        off += index[i] * strides[i];
    }
    return off;
}

Tensor Tensor::reshape(Shape new_shape) const {
    if (new_shape.numel() != numel()) {
        throw std::invalid_argument("can't reshape " + shape_.to_string() + " to " +
                                    new_shape.to_string());
    }
    Tensor view = *this;  // shares storage_
    view.shape_ = std::move(new_shape);
    return view;
}

Tensor Tensor::clone() const {
    Tensor copy(shape_, dtype_);
    if (nbytes()) std::memcpy(copy.storage_->data(), storage_->data(), nbytes());
    return copy;
}

}  // namespace tinyinfer
