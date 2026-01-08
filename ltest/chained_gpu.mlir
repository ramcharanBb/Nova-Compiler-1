module {
  // Debug Full Chain (Small 64x64 to separate hang from slowness)
  func.func @main1() -> tensor<64x64xf32, #nova.device<"1">> attributes { llvm.emit_c_interface } {
    
    // Constant
    %cst = arith.constant dense<1.0> : tensor<64x64xf32, #nova.device<"0">>
    
    // Copy H2D
    %dev0 = nova.to_device %cst : tensor<64x64xf32, #nova.device<"0">> -> tensor<64x64xf32, #nova.device<"1">>
    
    // Full Chain: Add -> Matmul -> Add
    %0 = nova.add %dev0, %dev0 : tensor<64x64xf32, #nova.device<"1">>, tensor<64x64xf32, #nova.device<"1">>
    %1 = nova.matmul %0, %dev0 : tensor<64x64xf32, #nova.device<"1">>, tensor<64x64xf32, #nova.device<"1">>
    %2 = nova.add %1, %0 : tensor<64x64xf32, #nova.device<"1">>, tensor<64x64xf32, #nova.device<"1">>
    
    return %2 : tensor<64x64xf32, #nova.device<"1">>
  }
}
