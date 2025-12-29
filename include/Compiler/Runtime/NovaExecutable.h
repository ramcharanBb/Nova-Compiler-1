//===- NovaExecutable.h - Executable for Nova Runtime ----------*- C++ -*-===//
//
// Nova Runtime - PJRT-compatible Executable
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_EXECUTABLE_H
#define NOVA_RUNTIME_EXECUTABLE_H

#include "NovaBuffer.h"
#include "NovaDevice.h"
#include <vector>
#include <memory>
#include <string>

namespace mlir {
namespace nova {
namespace runtime {

// Forward declaration
class NovaThunk;

/// Compiled executable (AOT)
class NovaExecutable {
public:
  /// Execute the program with given inputs
  std::vector<std::unique_ptr<NovaBuffer>> execute(
    const std::vector<NovaBuffer*>& inputs
  );
  
  /// Metadata
  std::string name() const { return name_; }
  NovaDevice* device() const { return device_; }
  
  /// Buffer requirements (from buffer assignment)
  size_t requiredScratchBytes() const { return scratch_bytes_; }
  
  /// Destructor
  ~NovaExecutable();
  
private:
  friend class NovaClient;
  friend class CPUBackend;
  friend class GPUBackend;
  
  /// Private constructor
  NovaExecutable(const std::string& name, NovaDevice* device);
  
  std::string name_;
  NovaDevice* device_;
  std::vector<std::unique_ptr<NovaThunk>> thunk_sequence_;
  size_t scratch_bytes_;
  
  /// Scratch memory pool
  void* scratch_buffer_;
};

} // namespace runtime
} // namespace nova
} // namespace mlir

#endif // NOVA_RUNTIME_EXECUTABLE_H