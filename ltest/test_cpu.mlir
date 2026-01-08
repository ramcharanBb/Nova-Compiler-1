module {
  func.func @test_cpu(%arg0: tensor<8x8xf32, #nova.device<"0">>, %arg1: tensor<8x8xf32, #nova.device<"0">>) 
  -> tensor<8x8xf32, #nova.device<"0">> {
    %10 = nova.transpose %arg0 : tensor<8x8xf32, #nova.device<"0">>
    return %10 : tensor<8x8xf32, #nova.device<"0">>
  }
}
