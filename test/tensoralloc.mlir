// RUN: nova-opt %s | FileCheck %s

// CHECK-LABEL: func @mixed_device_execution
func.func @mixed_device_execution(%a_cpu: tensor<10x10xf32, #nova.device<"cpu":0>>, 
                                  %b_cpu: tensor<10x10xf32, #nova.device<"cpu":0>>) 
                                  -> tensor<10x10xf32, #nova.device<"cpu":0>> {
  
  // 1. CPU Addition
  // CHECK: nova.add
  %sum_cpu = nova.add %a_cpu, %b_cpu : tensor<10x10xf32, #nova.device<"cpu":0>>, tensor<10x10xf32, #nova.device<"cpu":0>> -> tensor<10x10xf32, #nova.device<"cpu":0>>

  // 2. Move to GPU
  // CHECK: nova.to_device
  // CHECK-SAME: device = #nova.device<"cuda":0>
  %sum_gpu = nova.to_device %sum_cpu {device = #nova.device<"cuda":0>} : tensor<10x10xf32, #nova.device<"cpu":0>> -> tensor<10x10xf32, #nova.device<"cuda":0>>
  
  // 3. Create another GPU tensor (simulated for this test by moving b_cpu)
  %b_gpu = nova.to_device %b_cpu {device = #nova.device<"cuda":0>} : tensor<10x10xf32, #nova.device<"cpu":0>> -> tensor<10x10xf32, #nova.device<"cuda":0>>

  // 4. GPU Matmul
  // CHECK: nova.matmul
  %result_gpu = nova.matmul %sum_gpu, %b_gpu : tensor<10x10xf32, #nova.device<"cuda":0>>, tensor<10x10xf32, #nova.device<"cuda":0>> -> tensor<10x10xf32, #nova.device<"cuda":0>>

  // 5. Move back to CPU
  // CHECK: nova.to_device
  // CHECK-SAME: device = #nova.device<"cpu":0>
  %result_cpu = nova.to_device %result_gpu {device = #nova.device<"cpu":0>} : tensor<10x10xf32, #nova.device<"cuda":0>> -> tensor<10x10xf32, #nova.device<"cpu":0>>

  return %result_cpu : tensor<10x10xf32, #nova.device<"cpu":0>>
}