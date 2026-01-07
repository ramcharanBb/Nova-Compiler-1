module {
  func.func @test_cpu(%arg0: tensor<8x8xf32, #nova.device<"0">>, %arg1: tensor<8x8xf32, #nova.device<"0">>) 
  -> tensor<8x8xf32, #nova.device<"0">> {
    %0 = nova.add %arg0, %arg1 : tensor<8x8xf32, #nova.device<"0">>, tensor<8x8xf32, #nova.device<"0">>
    %1 = nova.matmul %0, %arg1 : tensor<8x8xf32, #nova.device<"0">>, tensor<8x8xf32, #nova.device<"0">>
    return %1 : tensor<8x8xf32, #nova.device<"0">>
  }
}
