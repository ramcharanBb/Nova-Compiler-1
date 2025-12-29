//===- DeviceTest.cpp - Device Unit Tests ----------------------*- C++ -*-===//
//
// Nova Runtime - Device Tests
//
//===----------------------------------------------------------------------===//

#include "Compiler/Runtime/NovaDevice.h"
#include <iostream>
#include <cassert>

using namespace mlir::nova::runtime;

void testDeviceEnumeration() {
  std::cout << "Testing device enumeration..." << std::endl;
  
  auto devices = NovaDevice::enumerateDevices();
  
  // Should have at least one CPU device
  assert(devices.size() >= 1 && "Should have at least one device");
  
  // First device should be CPU
  assert(devices[0]->description().kind() == DeviceKind::CPU && 
         "First device should be CPU");
  
  std::cout << "Found " << devices.size() << " device(s):" << std::endl;
  for (const auto& device : devices) {
    std::cout << "  - " << device->description().toString() << std::endl;
  }
  
  std::cout << "✓ Device enumeration test passed" << std::endl;
}

void testMemoryAllocation() {
  std::cout << "\nTesting memory allocation..." << std::endl;
  
  auto devices = NovaDevice::enumerateDevices();
  NovaDevice* cpuDevice = devices[0].get();
  
  // Allocate 1MB
  size_t size = 1024 * 1024;
  void* ptr = cpuDevice->allocate(size);
  assert(ptr != nullptr && "Allocation should succeed");
  
  // Deallocate
  cpuDevice->deallocate(ptr);
  
  std::cout << "✓ Memory allocation test passed" << std::endl;
}

void testMemoryCopy() {
  std::cout << "\nTesting memory copy..." << std::endl;
  
  auto devices = NovaDevice::enumerateDevices();
  NovaDevice* cpuDevice = devices[0].get();
  
  // Create test data
  const int N = 100;
  float hostData[N];
  for (int i = 0; i < N; ++i) {
    hostData[i] = static_cast<float>(i);
  }
  
  // Allocate device memory
  void* devicePtr = cpuDevice->allocate(N * sizeof(float));
  
  // Copy to device
  cpuDevice->copyHostToDevice(devicePtr, hostData, N * sizeof(float));
  
  // Copy back
  float resultData[N];
  cpuDevice->copyDeviceToHost(resultData, devicePtr, N * sizeof(float));
  
  // Verify
  for (int i = 0; i < N; ++i) {
    assert(resultData[i] == hostData[i] && "Data mismatch");
  }
  
  cpuDevice->deallocate(devicePtr);
  
  std::cout << "✓ Memory copy test passed" << std::endl;
}

int main() {
  std::cout << "=== Nova Runtime Device Tests ===" << std::endl;
  
  try {
    testDeviceEnumeration();
    testMemoryAllocation();
    testMemoryCopy();
    
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed: " << e.what() << std::endl;
    return 1;
  }
}