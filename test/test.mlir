module {
  func.func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<f32> {
    %0 = nova.matmul %arg0, %arg1 : tensor<8x16xf32>, tensor<16x10xf32>
    %1 = nova.add %0, %arg2 : tensor<8x10xf32>, tensor<1x10xf32>
    %2 = nova.reduce<sum> %1 : tensor<8x10xf32>
    return %2 : tensor<f32>
  }
}