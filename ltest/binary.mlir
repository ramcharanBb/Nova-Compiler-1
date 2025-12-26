module {
  func.func @binary_ops(
    %arg0: tensor<1x8xf32>,
    %arg1: tensor<1x8xf32>
  ) -> tensor<1x8xf32> {

    %0 = nova.add %arg0, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %1 = nova.sub %0, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %2 = nova.mul %1, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %3 = nova.div %2, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %4 = nova.mod %3, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %5 = nova.max %4, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %6 = nova.min %5, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %7 = nova.and %6, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %8 = nova.or  %arg0, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>
    %9 = nova.xor %arg0, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>

    return %arg0 : tensor<1x8xf32>
  }
}
