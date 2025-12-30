//===- NovaThunk.h - Thunk Abstraction for Nova Runtime --------*- C++ -*-===//
//
// Nova Runtime - Thunk Execution Units (Stub for Milestone 1)
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_THUNK_H
#define NOVA_RUNTIME_THUNK_H

#include <vector>
#include <memory>
#include <string>

namespace mlir {
namespace nova {
namespace runtime {

class NovaBuffer;
class NovaDevice;

/// Base class for all execution units
/// Full implementation will be in Milestone 2
class NovaThunk {
public:
  virtual ~NovaThunk() = default;
  
  /// Execute this thunk (stub for now)
  virtual void execute(
    const std::vector<NovaBuffer*>& buffers,
    void* scratchBuffer
  ) = 0;
  
  /// Debugging
  virtual std::string toString() const = 0;
  
protected:
  NovaDevice* device_;
};

} // namespace runtime
} // namespace nova
} // namespace mlir

#endif // NOVA_RUNTIME_THUNK_H