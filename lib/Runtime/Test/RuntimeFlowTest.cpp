//===- RuntimeFlowTest.cpp - Manual Execution Verification ----------------===//
//
// This file simulates what the compiler WOULD do.
// It manually builds a "Plan" and runs it to verify the Runtime stack.
//
//===----------------------------------------------------------------------===//

#include "Runtime/Core/HostContext.h"
#include "Runtime/Executor/ExecutionEngine.h"
#include "Runtime/Kernels/KernelRegistration.h"
#include <iostream>
#include <vector>
#include <cassert>

// Mock Tensor for the test (since we might not link against full TensorLib)
// In a real verification, we'd link against TensorLib.
struct MockTensor {
    float val;
    MockTensor(float v) : val(v) {}
    MockTensor& operator+(const MockTensor& other) {
        val += other.val;
        return *this;
    }
};

// We need to override the AddWrapper for this TEST to use MockTensor
// because HostKernels.cpp uses the real (or different mock) Tensor.
// So we'll register a "test.add" kernel here.
using namespace nova::runtime;

AsyncValue* TestAddWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
    auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
    auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
    
    MockTensor* lhs = static_cast<MockTensor*>(lhs_av->get());
    MockTensor* rhs = static_cast<MockTensor*>(rhs_av->get());
    
    std::cout << "  [KERNEL] Executing test.add: " << lhs->val << " + " << rhs->val << "\n";
    
    MockTensor* result = new MockTensor(lhs->val + rhs->val);
    return host->MakeAvailableAsyncValue<void*>(result);
}

int main() {
    std::cout << "=== Verification: Runtime Flow ===\n";

    // 1. Setup Host (Thread Pool)
    std::cout << "1. Creating HostContext...\n";
    HostContext host(4); // 4 threads

    // 2. Register Kernels
    // We register our test kernel to avoid linking complexities for this standalone test
    std::cout << "2. Registering Kernels...\n";
    KernelRegistry::Instance().RegisterKernel("test.add", Device::CPU, TestAddWrapper);

    // 3. Create Inputs
    std::cout << "3. Preparing Inputs (A=10.0, B=20.0)...\n";
    MockTensor* inputA = new MockTensor(10.0f);
    MockTensor* inputB = new MockTensor(20.0f);
    
    std::vector<void*> inputs = { inputA, inputB };

    // 4. Build Execution Plan (The "Recipe")
    // Graph: 
    //   Task 0: Add(Input0, Input1) -> Output
    std::cout << "4. Building Execution Plan...\n";
    RuntimeExecutionPlan plan;
    plan.output_task_id = 0; // The result of Task 0 is the graph output

    AsyncTask task0;
    task0.task_id = 0;
    task0.op_name = "test.add";
    task0.device = Device::CPU;
    
    // Args: Input[0], Input[1]
    task0.args.push_back(ArgInput{0});
    task0.args.push_back(ArgInput{1});
    
    plan.tasks.push_back(task0);

    // 5. Execute
    std::cout << "5. Executing Graph...\n";
    ExecutionEngine engine(&host);
    
    // We use ExecuteSync for simplicity in testing
    // In real usage, we'd get an AsyncValue back.
    AsyncValue* result_av = engine.Execute(plan, inputs, {});
    
    std::cout << "   ... Waiting for result ...\n";
    result_av->Await();

    // 6. Check Result
    if (result_av->IsError()) {
        std::cerr << "FAILED: " << result_av->GetError() << "\n";
        return 1;
    }

    auto* concrete_result = dynamic_cast<ConcreteAsyncValue<void*>*>(result_av);
    MockTensor* res = static_cast<MockTensor*>(concrete_result->get());
    
    std::cout << "6. Result: " << res->val << "\n";
    
    if (res->val == 30.0f) {
        std::cout << "SUCCESS! (10.0 + 20.0 = 30.0)\n";
    } else {
        std::cerr << "FAILURE: Expected 30.0, got " << res->val << "\n";
        return 1;
    }

    // Cleanup
    delete inputA;
    delete inputB;
    delete res;
    // Note: AsyncValues are ref-counted or managed by HostContext in real impl. 
    // Here we leak small amount for simple test.

    return 0;
}
