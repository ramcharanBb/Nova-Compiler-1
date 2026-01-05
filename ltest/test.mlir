//Testing file for bufferization process
module {
  func.func @test_bce(%arg0: tensor<8x8xf32, #nova.device<"1">>, %arg1: tensor<8x8xf32, #nova.device<"1">>) -> tensor<8x8xf32, #nova.device<"1">> {
    %1 = nova.matmul %arg0, %arg1 : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
   return %1 : tensor<8x8xf32, #nova.device<"1">>
  }
}


