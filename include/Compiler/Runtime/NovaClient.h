//===- NovaClient.h - Client for Nova Runtime ------------------*- C++ -*-===//
//
// Nova Runtime - PJRT-compatible Client
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_CLIENT_H
#define NOVA_RUNTIME_CLIENT_H

#include "NovaDevice.h"
#include "NovaBuffer.h"
#include "NovaExecutable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <vector>

namespace mlir {
namespace nova {
namespace runtime {

/// Compilation options
struct CompileOptions {
  DeviceKind targetDevice = DeviceKind::CPU;
  int deviceId = 0;
  bool enableFusion = true;
  bool enableBufferReuse = true;
  int optimizationLevel = 3; // 0-3
};

/// PJRT-compatible client
class NovaClient {
public:
  /// Factory method
  static std::unique_ptr<NovaClient> create();
  
  /// Destructor
  ~NovaClient();
  
  /// Device management
  std::vector<NovaDevice*> devices() const;
  NovaDevice* device(int id) const;
  NovaDevice* deviceByKind(DeviceKind kind, int index = 0) const;
  
  /// Compilation (AOT) - stub for now, will be implemented in Milestone 2
  std::unique_ptr<NovaExecutable> compile(
    mlir::ModuleOp module,
    const CompileOptions& options
  );
  
  /// Buffer creation
  std::unique_ptr<NovaBuffer> createBuffer(
    const void* data,
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device
  );
  
  /// Create uninitialized buffer
  std::unique_ptr<NovaBuffer> createUninitializedBuffer(
    const std::vector<int64_t>& shape,
    mlir::Type elementType,
    NovaDevice* device
  );
  
  /// Get MLIR context
  mlir::MLIRContext* getContext() { return context_.get(); }
  
private:
  /// Private constructor
  NovaClient();
  
  std::unique_ptr<mlir::MLIRContext> context_;
  std::vector<std::unique_ptr<NovaDevice>> devices_;
};

} // namespace runtime
} // namespace nova
} // namespace mlir

#endif // NOVA_RUNTIME_CLIENT_H