#pragma once

#include <cstddef>
#include <new>

namespace tinyinfer {

/// One cache line, and the width of an AVX-512 register: aligned loads/stores never split.
inline constexpr std::size_t kAlignment = 64;

/// Raw byte buffer aligned to kAlignment. Owns its memory, move-only, zero-initialized.
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t bytes) : size_(bytes) {
        if (bytes == 0) return;
        // round up so SIMD loops can read a full register past the last element safely
        const std::size_t padded = (bytes + kAlignment - 1) / kAlignment * kAlignment;
        data_ = static_cast<std::byte*>(::operator new(padded, std::align_val_t{kAlignment}));
        for (std::size_t i = 0; i < padded; ++i) data_[i] = std::byte{0};
    }
    ~AlignedBuffer() { release(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    [[nodiscard]] std::byte* data() noexcept { return data_; }
    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    void release() noexcept {
        if (data_) ::operator delete(data_, std::align_val_t{kAlignment});
        data_ = nullptr;
    }

    std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace tinyinfer
