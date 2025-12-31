//===- ExecutionEngine.cpp - Async Graph Executor -------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Executor/ExecutionEngine.h"
#include "Runtime/Executor/DependencyGraph.h"
#include "Runtime/Core/ErrorHandling.h"
#include <iostream>
#include <stdexcept>

namespace nova {
namespace runtime {

ExecutionEngine::ExecutionEngine(HostContext* host)
    : host_(host), registry_(&KernelRegistry::Instance()) {}

ExecutionEngine::~ExecutionEngine() = default;

AsyncValue* ExecutionEngine::Execute(
    const RuntimeExecutionPlan& plan,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  // Create intermediate AsyncValues for each task
  std::vector<AsyncValue*> intermediate_values(plan.tasks.size(), nullptr);
  
  // Create dependency graph if not cached
  DependencyGraph dep_graph;
  for (const auto& task : plan.tasks) {
    dep_graph.AddTask(task.task_id, task.dependencies);
  }
  
  // Dispatch all tasks
  for (const auto& task : plan.tasks) {
    // Create AsyncValue for this task's output
    intermediate_values[task.task_id] = host_->MakeUnconstructedAsyncValue<void>();
    
    // Get dependencies
    std::vector<AsyncValue*> deps;
    for (int dep_id : task.dependencies) {
      deps.push_back(intermediate_values[dep_id]);
    }
    
    // Dispatch task when dependencies are ready
    if (deps.empty()) {
      // No dependencies, dispatch immediately
      DispatchTask(task, intermediate_values, inputs, params);
    } else {
      // Wait for dependencies
      RunWhenReady(deps, [this, &task, &intermediate_values, &inputs, &params]() {
        DispatchTask(task, intermediate_values, inputs, params);
      });
    }
  }
  
  // Return the output AsyncValue
  return intermediate_values[plan.output_task_id];
}

void* ExecutionEngine::ExecuteSync(
    const RuntimeExecutionPlan& plan,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  auto* result = Execute(plan, inputs, params);
  result->Await();
  
  if (result->IsError()) {
    throw std::runtime_error("Execution failed: " + result->GetError());
  }
  
  // In real implementation, this would return Tensor*
  // For now, return nullptr as placeholder
  return nullptr;
}

void ExecutionEngine::DispatchTask(
    const AsyncTask& task,
    const std::vector<AsyncValue*>& intermediate_values,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  // Get the output AsyncValue for this task
  AsyncValue* output = intermediate_values[task.task_id];
  
  // Enqueue work on thread pool
  host_->EnqueueWork([this, &task, output, &intermediate_values, &inputs, &params]() {
    try {
      // Get the kernel for this operation
      KernelFunction kernel = registry_->GetKernel(task.op_name, task.device);
      
      // Resolve arguments
      std::vector<AsyncValue*> kernel_args;
      for (const auto& arg : task.args) {
        AsyncValue* resolved = ResolveArg(arg, intermediate_values, inputs, params);
        kernel_args.push_back(resolved);
      }
      
      // Execute kernel
      AsyncValue* result = kernel(kernel_args, host_);
      
      // For now, just mark output as complete
      // In real implementation, we'd transfer the result
      if (auto* concrete_output = dynamic_cast<ConcreteAsyncValue<void>*>(output)) {
        concrete_output->emplace();
      }
      
    } catch (const std::exception& e) {
      // Set error on output
      output->SetError(e.what());
    }
  });
}

void ExecutionEngine::RunWhenReady(
    const std::vector<AsyncValue*>& deps,
    std::function<void()> callback) {
  
  if (deps.empty()) {
    callback();
    return;
  }
  
  // Count remaining dependencies
  auto remaining = std::make_shared<std::atomic<int>>(deps.size());
  
  for (auto* dep : deps) {
    dep->AndThen([remaining, callback](AsyncValue* av) {
      if (av->IsError()) {
        // Propagate error - don't execute callback
        return;
      }
      
      int count = remaining->fetch_sub(1, std::memory_order_acq_rel);
      if (count == 1) {
        // All dependencies ready
        callback();
      }
    });
  }
}

AsyncValue* ExecutionEngine::ResolveArg(
    const TaskArg& arg,
    const std::vector<AsyncValue*>& intermediate_values,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  return std::visit([&](auto&& a) -> AsyncValue* {
    using T = std::decay_t<decltype(a)>;
    
    if constexpr (std::is_same_v<T, ArgInput>) {
      // Wrap input tensor in an available AsyncValue
      // In real implementation: return host_->MakeAvailableAsyncValue<Tensor>(*inputs[a.idx]);
      return host_->MakeAvailableAsyncValue<void*>(inputs[a.idx]);
      
    } else if constexpr (std::is_same_v<T, ArgParam>) {
      // Wrap param tensor in an available AsyncValue
      return host_->MakeAvailableAsyncValue<void*>(params[a.idx]);
      
    } else if constexpr (std::is_same_v<T, ArgSlot>) {
      // Return intermediate value
      return intermediate_values[a.slot];
      
    } else if constexpr (std::is_same_v<T, ArgLiteral>) {
      // Wrap literal in an available AsyncValue
      // Note: Passing nullptr for now as placeholder for literal data
      return host_->MakeAvailableAsyncValue<void*>(nullptr);
    }
    
    return nullptr;
  }, arg);
}

} // namespace runtime
} // namespace nova
