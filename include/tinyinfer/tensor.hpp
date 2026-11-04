#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyinfer/aligned.hpp"
#include "tinyinfer/shape.hpp"

namespace tinyinfer {

enum class DType : std::uint8_t { Float32, Int8, Int32, Int64 };

std::size_t dtype_size(DType t) noexcept;
const char* dtype_name(DType t) noexcept;

template <class T>
constexpr DType dtype_of();
template <>
constexpr DType dtype_of<float>() { return DType::Float32; }
template <>
constexpr DType dtype_of<std::int8_t>() { return DType::Int8; }
template <>
constexpr DType dtype_of<std::int32_t>() { return DType::Int32; }
template <>
constexpr DType dtype_of<std::int64_t>() { return DType::Int64; }

/// Dense, contiguous, row-major tensor on 64-byte aligned memory.
///
/// Copies are cheap and share storage (like a shared_ptr); `reshape` returns a view of the same
/// bytes. Use `clone` for an independent copy. Kept deliberately small: the engine plans buffers
/// itself, so tensors don't need strides or views into the middle of other tensors.
class Tensor {
public:
    Tensor() = default;
    Tensor(Shape shape, DType dtype);

    template <class T>
    static Tensor from(Shape shape, const std::vector<T>& values) {
        Tensor t(std::move(shape), dtype_of<T>());
        if (static_cast<std::int64_t>(values.size()) != t.numel()) {
            throw std::invalid_argument("got " + std::to_string(values.size()) +
                                        " values for shape " + t.shape().to_string());
        }
        T* dst = t.data<T>();
        for (std::size_t i = 0; i < values.size(); ++i) dst[i] = values[i];
        return t;
    }

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] std::int64_t numel() const noexcept { return shape_.numel(); }
    [[nodiscard]] std::size_t nbytes() const noexcept {
        return static_cast<std::size_t>(numel()) * dtype_size(dtype_);
    }

    template <class T>
    [[nodiscard]] T* data() {
        check_type(dtype_of<T>());
        return reinterpret_cast<T*>(storage_ ? storage_->data() : nullptr);
    }
    template <class T>
    [[nodiscard]] const T* data() const {
        check_type(dtype_of<T>());
        return reinterpret_cast<const T*>(storage_ ? storage_->data() : nullptr);
    }

    /// Element at a multi-dimensional index (bounds-checked; for tests and debugging).
    template <class T>
    [[nodiscard]] T at(const std::vector<std::int64_t>& index) const {
        return data<T>()[offset_of(index)];
    }

    /// Same data, new shape with the same number of elements. Shares storage.
    [[nodiscard]] Tensor reshape(Shape new_shape) const;
    [[nodiscard]] Tensor clone() const;
    [[nodiscard]] bool shares_storage_with(const Tensor& o) const noexcept {
        return storage_ && storage_ == o.storage_;
    }

private:
    void check_type(DType wanted) const;
    [[nodiscard]] std::int64_t offset_of(const std::vector<std::int64_t>& index) const;

    Shape shape_;
    DType dtype_ = DType::Float32;
    std::shared_ptr<AlignedBuffer> storage_;
};

}  // namespace tinyinfer
