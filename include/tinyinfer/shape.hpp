#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace tinyinfer {

/// Tensor dimensions, outermost first (NCHW for images).
class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<std::int64_t> dims);
    explicit Shape(std::vector<std::int64_t> dims);

    [[nodiscard]] std::size_t rank() const noexcept { return dims_.size(); }
    [[nodiscard]] std::int64_t operator[](std::size_t i) const { return dims_.at(i); }
    [[nodiscard]] const std::vector<std::int64_t>& dims() const noexcept { return dims_; }
    /// Number of elements; 1 for a scalar (rank 0).
    [[nodiscard]] std::int64_t numel() const noexcept;
    /// Row-major strides in elements, e.g. {2,3,4} -> {12,4,1}.
    [[nodiscard]] std::vector<std::int64_t> contiguous_strides() const;
    [[nodiscard]] std::string to_string() const;

    bool operator==(const Shape& o) const noexcept { return dims_ == o.dims_; }
    bool operator!=(const Shape& o) const noexcept { return !(*this == o); }

private:
    std::vector<std::int64_t> dims_;
};

}  // namespace tinyinfer
