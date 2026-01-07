module {
  func.func @test_bce(%arg0: tensor<8x8xf32, #nova.device<"1">>, %arg1: tensor<8x8xf32, #nova.device<"1">>) -> tensor<8x8xf32, #nova.device<"1">> {
    %1 = nova.mae %arg0, %arg1 : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    %2 = nova.mse %arg0, %arg1 : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    %3 = nova.bce %arg0, %arg1 : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    %4 = nova.cce %arg0, %arg1 : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    return %1 : tensor<8x8xf32, #nova.device<"1">>
  }
}
