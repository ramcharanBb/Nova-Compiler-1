module {
  func.func @binary_ops(
    %arg0: tensor<1x8xf32, #nova.device<"1">>,%arg1: tensor<1x8xf32, #nova.device<"1">>) -> tensor<1x8xf32, #nova.device<"1">> {

    %0 = nova.add %arg0, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %1 = nova.sub %0, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %2 = nova.mul %1, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %3 = nova.div %2, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %31=nova.mod %1 ,%arg0: tensor<1x8xf32, #nova.device<"1">>,tensor<1x8xf32, #nova.device<"1">>
    %32=nova.pow %1,%arg0: tensor<1x8xf32, #nova.device<"1">>,tensor<1x8xf32, #nova.device<"1">>
    %4 = nova.mod %3, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %5 = nova.max %4, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %6 = nova.min %5, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %7 = nova.and %6, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %8 = nova.or  %arg0, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>
    %9 = nova.xor %arg0, %arg1 : tensor<1x8xf32, #nova.device<"1">>, tensor<1x8xf32, #nova.device<"1">>

    return %arg0 : tensor<1x8xf32, #nova.device<"1">>
  }
}
