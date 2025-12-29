//===- NovaBuffer.h - Buffer Abstraction for Nova Runtime ------*- C++ -*-===//
//
// Nova Runtime - PJRT-compatible Buffer Abstraction
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_BUFFER_H
#define NOVA_RUNTIME_BUFFER_H

#include "NovaDevice.h"
#include "mlir/IR/BuiltinTypes.h"
#include <vector>
#include <memory>

namespace mlir {
namespace nova {
namespace runtime {

/// Buffer layout information
struct BufferLayout {
  std::vector<int64_t> dimensions;
  std::vector<int64_t> strides;
  
  /// Check if layout is row-major (C-style)
  bool isRowMajor() const;
  
  /// Calculate linear index from multi-dimensional index
  size_t linearIndex(const std::vector<int64_t>& indices) const;
};

/// PJRT-compatible buffer
class NovaBuffer {
public:
  /// Create buffer from host data
  static std::unique_ptr<NovaBuffer> createFromHost(
    const void* data,
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device
  );
  
  /// Create uninitialized buffer on device
  static std::unique_ptr<NovaBuffer> createUninitialized(
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device
  );
  
  /// Destructor - frees device memory
  ~NovaBuffer();
  
  /// Data access (blocking operations)
  void copyToHost(void* dst) const;
  void copyFromHost(const void* src);
  
  /// Metadata accessors
  const std::vector<int64_t>& shape() const { return layout_.dimensions; }
  const BufferLayout& layout() const { return layout_; }
  mlir::Type elementType() const { return element_type_; }
  NovaDevice* device() const { return device_; }
  size_t sizeInBytes() const { return size_bytes_; }
  
  /// Device pointer (for execution)
  void* devicePointer() const { return device_ptr_; }
  
  /// Element size in bytes
  size_t elementSizeBytes() const;
  
  /// Total number of elements
  size_t numElements() const;
  
private:
  /// Private constructor
  NovaBuffer(NovaDevice* device, void* devicePtr,
             const BufferLayout& layout, mlir::Type elementType,
             size_t sizeBytes);
  
  NovaDevice* device_;
  void* device_ptr_;
  BufferLayout layout_;
  mlir::Type element_type_;
  size_t size_bytes_;
};

} // namespace runtime
} // namespace nova
} // namespace mlir

#endif // NOVA_RUNTIME_BUFFER_H