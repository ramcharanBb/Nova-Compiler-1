//===- NovaDevice.h - Device Abstraction for Nova Runtime ------*- C++ -*-===//
//
// Nova Runtime - PJRT-compatible Device Abstraction
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_DEVICE_H
#define NOVA_RUNTIME_DEVICE_H

#include <string>
#include <vector>
#include <memory>
#include <cstddef>

namespace mlir {
namespace nova {
namespace runtime {

/// Device kind enumeration
enum class DeviceKind {
  CPU,
  CUDA
};

/// Device description (immutable properties)
class NovaDeviceDescription {
public:
  NovaDeviceDescription(DeviceKind kind, int deviceId, 
                       size_t totalMemory, int computeCapability)
      : kind_(kind), device_id_(deviceId), 
        total_memory_(totalMemory), 
        compute_capability_(computeCapability) {}

  DeviceKind kind() const { return kind_; }
  int deviceId() const { return device_id_; }
  std::string toString() const;
  
  // Device capabilities
  size_t totalMemoryBytes() const { return total_memory_; }
  int computeCapability() const { return compute_capability_; }
  
private:
  DeviceKind kind_;
  int device_id_;
  size_t total_memory_;
  int compute_capability_;
};

/// Device (runtime instance)
class NovaDevice {
public:
  /// Enumerate all available devices (CPU + CUDA)
  static std::vector<std::unique_ptr<NovaDevice>> enumerateDevices();
  
  /// Constructor
  NovaDevice(const NovaDeviceDescription& description, bool isAddressable)
      : description_(description), is_addressable_(isAddressable) {}
  
  /// Get device description
  const NovaDeviceDescription& description() const { return description_; }
  
  /// Check if this device is addressable from this process
  bool isAddressable() const { return is_addressable_; }
  
  /// Memory management
  void* allocate(size_t bytes);
  void deallocate(void* ptr);
  size_t availableMemoryBytes() const;
  
  /// Copy operations
  void copyHostToDevice(void* devicePtr, const void* hostPtr, size_t bytes);
  void copyDeviceToHost(void* hostPtr, const void* devicePtr, size_t bytes);
  void copyDeviceToDevice(void* dstPtr, const void* srcPtr, size_t bytes);
  
private:
  NovaDeviceDescription description_;
  bool is_addressable_;
};

} // namespace runtime
} // namespace nova
} // namespace mlir

#endif // NOVA_RUNTIME_DEVICE_H