module {
  // Case 1: Chained GPU Operations
  // Expectation: 1 H2D copy, multiple kernels, 1 D2H copy (if returned)
  func.func @test_chained_gpu(%arg0: tensor<32x32xf32, #nova.device<"1">>) -> tensor<32x32xf32, #nova.device<"1">> {
    %0 = nova.add %arg0, %arg0 : tensor<32x32xf32, #nova.device<"1">>, tensor<32x32xf32, #nova.device<"1">>
    %1 = nova.matmul %0, %arg0 : tensor<32x32xf32, #nova.device<"1">>, tensor<32x32xf32, #nova.device<"1">>
    %2 = nova.add %1, %0 : tensor<32x32xf32, #nova.device<"1">>, tensor<32x32xf32, #nova.device<"1">>
    return %2 : tensor<32x32xf32, #nova.device<"1">>
  }

  // Case 2: Ping-Pong Execution (Host -> Device -> Host -> Device)
  // Expectation: Multiple specialized copies (H2D, D2H, H2D)
  // Note: This effectively tests if the compiler inserts copies correctly between ops
  func.func @test_ping_pong(%arg0: tensor<32x32xf32, #nova.device<"0">>) -> tensor<32x32xf32, #nova.device<"1">> {
    // Host -> Device
    %dev0 = nova.to_device %arg0 : tensor<32x32xf32, #nova.device<"0">> -> tensor<32x32xf32, #nova.device<"1">>
    // GPU Op
    %dev1 = nova.add %dev0, %dev0 : tensor<32x32xf32, #nova.device<"1">>, tensor<32x32xf32, #nova.device<"1">>
    // Device -> Host
    %host0 = nova.to_device %dev1 : tensor<32x32xf32, #nova.device<"1">> -> tensor<32x32xf32, #nova.device<"0">>
    // CPU Op
    %host1 = nova.add %host0, %host0 : tensor<32x32xf32, #nova.device<"0">>, tensor<32x32xf32, #nova.device<"0">>
    // Host -> Device
    %dev2 = nova.to_device %host1 : tensor<32x32xf32, #nova.device<"0">> -> tensor<32x32xf32, #nova.device<"1">>
    return %dev2 : tensor<32x32xf32, #nova.device<"1">>
  }

  // Case 3: Matmul on GPU (Heavy Compute)
  func.func @test_gpu_matmul(%arg0: tensor<128x128xf32, #nova.device<"1">>, %arg1: tensor<128x128xf32, #nova.device<"1">>) -> tensor<128x128xf32, #nova.device<"1">> {
    %0 = nova.matmul %arg0, %arg1 : tensor<128x128xf32, #nova.device<"1">>, tensor<128x128xf32, #nova.device<"1">>
    return %0 : tensor<128x128xf32, #nova.device<"1">>
  }

  // Case 4: CPU Broadcast/Constant interaction
  func.func @test_cpu_constant_interaction() -> tensor<16x16xf32, #nova.device<"0">> {
    %cst = arith.constant dense<1.0> : tensor<16x16xf32, #nova.device<"0">>
    %0 = nova.add %cst, %cst : tensor<16x16xf32, #nova.device<"0">>, tensor<16x16xf32, #nova.device<"0">>
    return %0 : tensor<16x16xf32, #nova.device<"0">>
  }
}
