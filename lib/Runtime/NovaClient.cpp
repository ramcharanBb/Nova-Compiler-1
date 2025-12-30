//===- NovaClient.cpp - Client Implementation ------------------*- C++ -*-===//
//
// Nova Runtime - Client Implementation
//
//===----------------------------------------------------------------------===//

#include "Compiler/Runtime/NovaClient.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <stdexcept>

namespace mlir {
namespace nova {
namespace runtime {

//===----------------------------------------------------------------------===//
// NovaClient - Construction
//===----------------------------------------------------------------------===//

NovaClient::NovaClient() {
  // Initialize MLIR context
  context_ = std::make_unique<mlir::MLIRContext>();
  
  // Load required dialects
  context_->loadDialect<mlir::arith::ArithDialect>();
  context_->loadDialect<mlir::func::FuncDialect>();
  context_->loadDialect<mlir::tensor::TensorDialect>();
  
  // Enumerate devices
  devices_ = NovaDevice::enumerateDevices();
}

NovaClient::~NovaClient() = default;

std::unique_ptr<NovaClient> NovaClient::create() {
  return std::unique_ptr<NovaClient>(new NovaClient());
}

//===----------------------------------------------------------------------===//
// NovaClient - Device Management
//===----------------------------------------------------------------------===//

std::vector<NovaDevice*> NovaClient::devices() const {
  std::vector<NovaDevice*> result;
  for (const auto& device : devices_) {
    result.push_back(device.get());
  }
  return result;
}

NovaDevice* NovaClient::device(int id) const {
  if (id < 0 || id >= static_cast<int>(devices_.size())) {
    throw std::runtime_error("Invalid device ID");
  }
  return devices_[id].get();
}

NovaDevice* NovaClient::deviceByKind(DeviceKind kind, int index) const {
  int count = 0;
  for (const auto& device : devices_) {
    if (device->description().kind() == kind) {
      if (count == index) {
        return device.get();
      }
      count++;
    }
  }
  throw std::runtime_error("Device not found");
}

//===----------------------------------------------------------------------===//
// NovaClient - Compilation (Stub)
//===----------------------------------------------------------------------===//

std::unique_ptr<NovaExecutable> NovaClient::compile(
    mlir::ModuleOp module,
    const CompileOptions& options) {
  
  // Stub implementation for Milestone 1
  // Will be fully implemented in Milestone 2 with backend integration
  throw std::runtime_error("NovaClient::compile() not yet implemented - will be added in Milestone 2");
}

//===----------------------------------------------------------------------===//
// NovaClient - Buffer Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<NovaBuffer> NovaClient::createBuffer(
    const void* data,
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device) {
  
  return NovaBuffer::createFromHost(data, shape, elementType, device);
}

std::unique_ptr<NovaBuffer> NovaClient::createUninitializedBuffer(
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device) {
  
  return NovaBuffer::createUninitialized(shape, elementType, device);
}

} // namespace runtime
} // namespace nova
} // namespace mlir