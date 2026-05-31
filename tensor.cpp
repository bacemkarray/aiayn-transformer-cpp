#include "tensor.hpp"

#include <stdexcept>
#include <utility>

Tensor::Tensor()
    : storage_(std::make_shared<std::vector<value_type>>()),
      storage_offset_(0) {}

Tensor::Tensor(Shape shape)
    : shape_(std::move(shape)),
      strides_(make_contiguous_strides(shape_)),
      storage_(std::make_shared<std::vector<value_type>>(compute_numel(shape_))),
      storage_offset_(0) {}

Tensor::Tensor(Shape shape, std::vector<value_type> values)
    : shape_(std::move(shape)),
      strides_(make_contiguous_strides(shape_)),
      storage_(std::make_shared<std::vector<value_type>>(std::move(values))),
      storage_offset_(0) {
    if (storage_->size() != compute_numel(shape_)) {
        throw std::invalid_argument("Tensor values do not match shape");
    }
}

Tensor::Tensor(Shape shape,
               Strides strides,
               std::shared_ptr<std::vector<value_type>> storage,
               size_type storage_offset)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      storage_(std::move(storage)),
      storage_offset_(storage_offset) {
    require_valid_storage();
}

Tensor::size_type Tensor::ndim() const noexcept {
    return shape_.size();
}

Tensor::size_type Tensor::numel() const noexcept {
    return compute_numel(shape_);
}

bool Tensor::empty() const noexcept {
    return numel() == 0;
}

const Tensor::Shape& Tensor::shape() const noexcept {
    return shape_;
}

const Tensor::Strides& Tensor::strides() const noexcept {
    return strides_;
}

Tensor::size_type Tensor::storage_offset() const noexcept {
    return storage_offset_;
}

Tensor::value_type* Tensor::data() noexcept {
    return storage_->data() + storage_offset_;
}

const Tensor::value_type* Tensor::data() const noexcept {
    return storage_->data() + storage_offset_;
}

Tensor::value_type& Tensor::operator[](size_type index) {
    if (index >= numel()) {
        throw std::out_of_range("Tensor index out of range");
    }

    return data()[index];
}

const Tensor::value_type& Tensor::operator[](size_type index) const {
    if (index >= numel()) {
        throw std::out_of_range("Tensor index out of range");
    }

    return data()[index];
}

Tensor Tensor::squeeze() const {
    throw std::logic_error("Tensor::squeeze() is not implemented yet");
}

Tensor Tensor::squeeze(size_type dim) const {
    (void)dim;
    throw std::logic_error("Tensor::squeeze(dim) is not implemented yet");
}

Tensor Tensor::unsqueeze(size_type dim) const {
    (void)dim;
    throw std::logic_error("Tensor::unsqueeze(dim) is not implemented yet");
}

Tensor Tensor::reshape(Shape new_shape) const {
    (void)new_shape;
    throw std::logic_error("Tensor::reshape(new_shape) is not implemented yet");
}

Tensor Tensor::view(Shape new_shape) const {
    (void)new_shape;
    throw std::logic_error("Tensor::view(new_shape) is not implemented yet");
}

Tensor::size_type Tensor::compute_numel(const Shape& shape) {
    if (shape.empty()) {
        return 0;
    }

    size_type total = 1;
    for (size_type dim : shape) {
        total *= dim;
    }

    return total;
}

Tensor::Strides Tensor::make_contiguous_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);

    for (size_type i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }

    return strides;
}

void Tensor::require_valid_storage() const {
    if (!storage_) {
        throw std::invalid_argument("Tensor storage cannot be null");
    }

    if (storage_offset_ > storage_->size()) {
        throw std::out_of_range("Tensor storage offset is out of range");
    }
}
