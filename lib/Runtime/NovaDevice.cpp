//===- NovaDevice.cpp - Device Implementation ------------------*- C++ -*-===//
//
// Nova Runtime - Device Implementation
//
//===----------------------------------------------------------------------===//

#include "Compiler/Runtime/NovaDevice.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

#ifdef NOVA_ENABLE_CUDA
#include <cuda.h>
#endif

namespace mlir {
namespace nova {
namespace runtime {

//===----------------------------------------------------------------------===//
// NovaDeviceDescription
//===----------------------------------------------------------------------===//

std::string NovaDeviceDescription::toString() const {
  std::ostringstream oss;
  if (kind_ == DeviceKind::CPU) {
    oss << "CPU Device " << device_id_;
  } else if (kind_ == DeviceKind::CUDA) {
    oss << "CUDA Device " << device_id_ 
        << " (Compute Capability: " << compute_capability_ << ")";
  }
  oss << ", Memory: " << (total_memory_ / (1024 * 1024)) << " MB";
  return oss.str();
}

//===----------------------------------------------------------------------===//
// NovaDevice - Device Enumeration
//===----------------------------------------------------------------------===//

std::vector<std::unique_ptr<NovaDevice>> NovaDevice::enumerateDevices() {
  std::vector<std::unique_ptr<NovaDevice>> devices;
  
  // Always add CPU device
  {
    // Get system memory (simplified - assumes 8GB)
    size_t cpuMemory = 8ULL * 1024 * 1024 * 1024; // 8 GB
    NovaDeviceDescription cpuDesc(DeviceKind::CPU, 0, cpuMemory, 0);
    devices.push_back(std::make_unique<NovaDevice>(cpuDesc, true));
  }
  
#ifdef NOVA_ENABLE_CUDA
  // Initialize CUDA
  CUresult result = cuInit(0);
  if (result == CUDA_SUCCESS) {
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
      CUdevice cuDevice;
      if (cuDeviceGet(&cuDevice, i) == CUDA_SUCCESS) {
        // Get device properties
        size_t totalMem = 0;
        cuDeviceTotalMem(&totalMem, cuDevice);
        
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
        int computeCap = major * 10 + minor;
        
        NovaDeviceDescription cudaDesc(DeviceKind::CUDA, i, totalMem, computeCap);
        devices.push_back(std::make_unique<NovaDevice>(cudaDesc, true));
      }
    }
  }
#endif
  
  return devices;
}

//===----------------------------------------------------------------------===//
// NovaDevice - Memory Management
//===----------------------------------------------------------------------===//

void* NovaDevice::allocate(size_t bytes) {
  if (description_.kind() == DeviceKind::CPU) {
    // CPU allocation - use aligned allocation for better performance
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(bytes, 64); // 64-byte alignment
#else
    if (posix_memalign(&ptr, 64, bytes) != 0) {
      ptr = nullptr;
    }
#endif
    if (!ptr) {
      throw std::runtime_error("Failed to allocate CPU memory");
    }
    return ptr;
  }
  
#ifdef NOVA_ENABLE_CUDA
  if (description_.kind() == DeviceKind::CUDA) {
    CUdeviceptr devicePtr;
    CUresult result = cuMemAlloc(&devicePtr, bytes);
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error("Failed to allocate CUDA memory");
    }
    return reinterpret_cast<void*>(devicePtr);
  }
#endif
  
  throw std::runtime_error("Unsupported device kind");
}

void NovaDevice::deallocate(void* ptr) {
  if (!ptr) return;
  
  if (description_.kind() == DeviceKind::CPU) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    return;
  }
  
#ifdef NOVA_ENABLE_CUDA
  if (description_.kind() == DeviceKind::CUDA) {
    cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
    return;
  }
#endif
}

size_t NovaDevice::availableMemoryBytes() const {
  if (description_.kind() == DeviceKind::CPU) {
    // Simplified - return total memory
    return description_.totalMemoryBytes();
  }
  
#ifdef NOVA_ENABLE_CUDA
  if (description_.kind() == DeviceKind::CUDA) {
    size_t free = 0, total = 0;
    cuMemGetInfo(&free, &total);
    return free;
  }
#endif
  
  return 0;
}

//===----------------------------------------------------------------------===//
// NovaDevice - Copy Operations
//===----------------------------------------------------------------------===//

void NovaDevice::copyHostToDevice(void* devicePtr, const void* hostPtr, size_t bytes) {
  if (description_.kind() == DeviceKind::CPU) {
    std::memcpy(devicePtr, hostPtr, bytes);
    return;
  }
  
#ifdef NOVA_ENABLE_CUDA
  if (description_.kind() == DeviceKind::CUDA) {
    CUresult result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(devicePtr), 
                                    hostPtr, bytes);
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error("CUDA host-to-device copy failed");
    }
    return;
  }
#endif
  
  throw std::runtime_error("Unsupported device kind");
}

void NovaDevice::copyDeviceToHost(void* hostPtr, const void* devicePtr, size_t bytes) {
  if (description_.kind() == DeviceKind::CPU) {
    std::memcpy(hostPtr, devicePtr, bytes);
    return;
  }
  
#ifdef NOVA_ENABLE_CUDA
  if (description_.kind() == DeviceKind::CUDA) {
    CUresult result = cuMemcpyDtoH(hostPtr, 
                                    reinterpret_cast<CUdeviceptr>(devicePtr), 
                                    bytes);
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error("CUDA device-to-host copy failed");
    }
    return;
  }
#endif
  
  throw std::runtime_error("Unsupported device kind");
}

void NovaDevice::copyDeviceToDevice(void* dstPtr, const void* srcPtr, size_t bytes) {
  if (description_.kind() == DeviceKind::CPU) {
    std::memcpy(dstPtr, srcPtr, bytes);
    return;
  }
  
#ifdef NOVA_ENABLE_CUDA
  if (description_.kind() == DeviceKind::CUDA) {
    CUresult result = cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dstPtr),
                                    reinterpret_cast<CUdeviceptr>(srcPtr),
                                    bytes);
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error("CUDA device-to-device copy failed");
    }
    return;
  }
#endif
  
  throw std::runtime_error("Unsupported device kind");
}

} // namespace runtime
} // namespace nova
} // namespace mlir