module {
  func.func @main1() -> tensor<2048x2048xf32, #nova.device<"0">> {
    %cst0 = arith.constant dense<1.0> : tensor<2048x2048xf32, #nova.device<"0">>
    %cst1 = arith.constant dense<2.0> : tensor<2048x2048xf32, #nova.device<"0">>
    %0 = nova.add %cst0, %cst1 : tensor<2048x2048xf32, #nova.device<"0">>, tensor<2048x2048xf32, #nova.device<"0">>
    %1 = nova.matmul %0, %cst1 : tensor<2048x2048xf32, #nova.device<"0">>, tensor<2048x2048xf32, #nova.device<"0">>
    return %1 : tensor<2048x2048xf32, #nova.device<"0">>
  }
}
