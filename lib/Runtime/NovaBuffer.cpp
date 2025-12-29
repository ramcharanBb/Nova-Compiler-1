//===- NovaBuffer.cpp - Buffer Implementation ------------------*- C++ -*-===//
//
// Nova Runtime - Buffer Implementation
//
//===----------------------------------------------------------------------===//

#include "Compiler/Runtime/NovaBuffer.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include <cstring>
#include <stdexcept>
#include <numeric>

namespace mlir {
namespace nova {
namespace runtime {

//===----------------------------------------------------------------------===//
// BufferLayout
//===----------------------------------------------------------------------===//

bool BufferLayout::isRowMajor() const {
  if (dimensions.empty()) return true;
  
  // Row-major: strides decrease from left to right
  // stride[i] = stride[i+1] * dimensions[i+1]
  for (size_t i = 0; i < strides.size() - 1; ++i) {
    if (strides[i] != strides[i + 1] * dimensions[i + 1]) {
      return false;
    }
  }
  return strides.back() == 1;
}

size_t BufferLayout::linearIndex(const std::vector<int64_t>& indices) const {
  if (indices.size() != dimensions.size()) {
    throw std::runtime_error("Index dimensionality mismatch");
  }
  
  size_t linearIdx = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] < 0 || indices[i] >= dimensions[i]) {
      throw std::runtime_error("Index out of bounds");
    }
    linearIdx += indices[i] * strides[i];
  }
  return linearIdx;
}

//===----------------------------------------------------------------------===//
// NovaBuffer - Helper Functions
//===----------------------------------------------------------------------===//

static size_t getElementSize(mlir::Type type) {
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type)) {
    return (intType.getWidth() + 7) / 8; // Round up to bytes
  }
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type)) {
    return floatType.getWidth() / 8;
  }
  throw std::runtime_error("Unsupported element type");
}

static BufferLayout createRowMajorLayout(const std::vector<int64_t>& shape) {
  BufferLayout layout;
  layout.dimensions = shape;
  layout.strides.resize(shape.size());
  
  if (shape.empty()) {
    return layout;
  }
  
  // Row-major: rightmost dimension has stride 1
  layout.strides.back() = 1;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    layout.strides[i] = layout.strides[i + 1] * shape[i + 1];
  }
  
  return layout;
}

//===----------------------------------------------------------------------===//
// NovaBuffer - Construction
//===----------------------------------------------------------------------===//

NovaBuffer::NovaBuffer(NovaDevice* device, void* devicePtr,
                       const BufferLayout& layout, mlir::Type elementType,
                       size_t sizeBytes)
    : device_(device), device_ptr_(devicePtr), layout_(layout),
      element_type_(elementType), size_bytes_(sizeBytes) {}

std::unique_ptr<NovaBuffer> NovaBuffer::createFromHost(
    const void* data,
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device) {
  
  // Calculate buffer size
  size_t numElements = 1;
  for (auto dim : shape) {
    numElements *= dim;
  }
  size_t elementSize = getElementSize(elementType);
  size_t totalBytes = numElements * elementSize;
  
  // Allocate device memory
  void* devicePtr = device->allocate(totalBytes);
  
  // Create layout
  BufferLayout layout = createRowMajorLayout(shape);
  
  // Create buffer
  auto buffer = std::unique_ptr<NovaBuffer>(
    new NovaBuffer(device, devicePtr, layout, elementType, totalBytes)
  );
  
  // Copy data to device
  if (data) {
    buffer->copyFromHost(data);
  }
  
  return buffer;
}

std::unique_ptr<NovaBuffer> NovaBuffer::createUninitialized(
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device) {
  
  return createFromHost(nullptr, shape, elementType, device);
}

NovaBuffer::~NovaBuffer() {
  if (device_ptr_) {
    device_->deallocate(device_ptr_);
  }
}

//===----------------------------------------------------------------------===//
// NovaBuffer - Data Transfer
//===----------------------------------------------------------------------===//

void NovaBuffer::copyToHost(void* dst) const {
  if (!dst) {
    throw std::runtime_error("Null destination pointer");
  }
  device_->copyDeviceToHost(dst, device_ptr_, size_bytes_);
}

void NovaBuffer::copyFromHost(const void* src) {
  if (!src) {
    throw std::runtime_error("Null source pointer");
  }
  device_->copyHostToDevice(device_ptr_, src, size_bytes_);
}

//===----------------------------------------------------------------------===//
// NovaBuffer - Metadata
//===----------------------------------------------------------------------===//

size_t NovaBuffer::elementSizeBytes() const {
  return getElementSize(element_type_);
}

size_t NovaBuffer::numElements() const {
  if (layout_.dimensions.empty()) {
    return 0;
  }
  return std::accumulate(layout_.dimensions.begin(), 
                        layout_.dimensions.end(),
                        1LL, std::multiplies<int64_t>());
}

} // namespace runtime
} // namespace nova
} // namespace mlir