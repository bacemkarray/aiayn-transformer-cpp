#pragma once

#include <cstddef>
#include <memory>
#include <vector>

class Tensor {
public:
    using value_type = float;
    using size_type = std::size_t;
    using Shape = std::vector<size_type>;
    using Strides = std::vector<size_type>;

    Tensor();
    explicit Tensor(Shape shape);
    Tensor(Shape shape, std::vector<value_type> values);

    size_type ndim() const noexcept;
    size_type numel() const noexcept;
    bool empty() const noexcept;

    const Shape& shape() const noexcept;
    const Strides& strides() const noexcept;
    size_type storage_offset() const noexcept;

    value_type* data() noexcept;
    const value_type* data() const noexcept;

    value_type& operator[](size_type index);
    const value_type& operator[](size_type index) const;

    Tensor squeeze() const;
    Tensor squeeze(size_type dim) const;
    Tensor unsqueeze(size_type dim) const;
    Tensor reshape(Shape new_shape) const;
    Tensor view(Shape new_shape) const;

private:
    Shape shape_;
    Strides strides_;
    std::shared_ptr<std::vector<value_type>> storage_;
    size_type storage_offset_;

    Tensor(Shape shape,
           Strides strides,
           std::shared_ptr<std::vector<value_type>> storage,
           size_type storage_offset);

    static size_type compute_numel(const Shape& shape);
    static Strides make_contiguous_strides(const Shape& shape);
    void require_valid_storage() const;
};
