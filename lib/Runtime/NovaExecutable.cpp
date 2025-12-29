//===- NovaExecutable.cpp - Executable Implementation ----------*- C++ -*-===//
//
// Nova Runtime - Executable Implementation
//
//===----------------------------------------------------------------------===//

#include "Compiler/Runtime/NovaExecutable.h"
#include "Compiler/Runtime/NovaThunk.h"
#include <stdexcept>

namespace mlir {
namespace nova {
namespace runtime {

NovaExecutable::NovaExecutable(const std::string& name, NovaDevice* device)
    : name_(name), device_(device), scratch_bytes_(0), scratch_buffer_(nullptr) {}

NovaExecutable::~NovaExecutable() {
  if (scratch_buffer_) {
    device_->deallocate(scratch_buffer_);
  }
}

std::vector<std::unique_ptr<NovaBuffer>> NovaExecutable::execute(
    const std::vector<NovaBuffer*>& inputs) {
  
  // Stub implementation for Milestone 1
  // Will be fully implemented in Milestone 2 with thunk execution
  throw std::runtime_error("NovaExecutable::execute() not yet implemented - will be added in Milestone 2");
}

} // namespace runtime
} // namespace nova
} // namespace mlir