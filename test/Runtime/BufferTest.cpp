//===- BufferTest.cpp - Buffer Unit Tests ----------------------*- C++ -*-===//
//
// Nova Runtime - Buffer Tests
//
//===----------------------------------------------------------------------===//

#include "Compiler/Runtime/NovaClient.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mlir::nova::runtime;

void testBufferCreation() {
  std::cout << "Testing buffer creation..." << std::endl;
  
  auto client = NovaClient::create();
  auto* cpuDevice = client->device(0);
  auto* context = client->getContext();
  
  // Create test data
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> shape = {2, 3};
  
  // Create buffer
  mlir::OpBuilder builder(context);
  auto buffer = client->createBuffer(
    data.data(), 
    shape, 
    builder.getF32Type(),
    cpuDevice
  );
  
  assert(buffer != nullptr && "Buffer creation should succeed");
  assert(buffer->shape() == shape && "Shape mismatch");
  assert(buffer->numElements() == 6 && "Element count mismatch");
  assert(buffer->sizeInBytes() == 6 * sizeof(float) && "Size mismatch");
  
  std::cout << "✓ Buffer creation test passed" << std::endl;
}

void testBufferCopy() {
  std::cout << "\nTesting buffer copy..." << std::endl;
  
  auto client = NovaClient::create();
  auto* cpuDevice = client->device(0);
  auto* context = client->getContext();
  
  // Create test data
  std::vector<float> data = {1.5f, 2.5f, 3.5f, 4.5f};
  std::vector<int64_t> shape = {4};
  
  // Create buffer and copy data
  mlir::OpBuilder builder(context);
  auto buffer = client->createBuffer(
    data.data(),
    shape,
    builder.getF32Type(),
    cpuDevice
  );
  
  // Copy back to host
  std::vector<float> result(4);
  buffer->copyToHost(result.data());
  
  // Verify
  for (size_t i = 0; i < data.size(); ++i) {
    assert(std::abs(result[i] - data[i]) < 1e-6f && "Data mismatch");
  }
  
  std::cout << "✓ Buffer copy test passed" << std::endl;
}

void testBufferLayout() {
  std::cout << "\nTesting buffer layout..." << std::endl;
  
  auto client = NovaClient::create();
  auto* cpuDevice = client->device(0);
  auto* context = client->getContext();
  
  std::vector<int64_t> shape = {3, 4, 5};
  std::vector<float> data(60, 1.0f);
  
  mlir::OpBuilder builder(context);
  auto buffer = client->createBuffer(
    data.data(),
    shape,
    builder.getF32Type(),
    cpuDevice
  );
  
  const auto& layout = buffer->layout();
  assert(layout.isRowMajor() && "Layout should be row-major");
  assert(layout.dimensions == shape && "Dimensions mismatch");
  
  // Check strides for row-major: [20, 5, 1]
  assert(layout.strides.size() == 3 && "Strides size mismatch");
  assert(layout.strides[0] == 20 && "Stride[0] mismatch");
  assert(layout.strides[1] == 5 && "Stride[1] mismatch");
  assert(layout.strides[2] == 1 && "Stride[2] mismatch");
  
  std::cout << "✓ Buffer layout test passed" << std::endl;
}

int main() {
  std::cout << "=== Nova Runtime Buffer Tests ===" << std::endl;
  
  try {
    testBufferCreation();
    testBufferCopy();
    testBufferLayout();
    
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed: " << e.what() << std::endl;
    return 1;
  }
}