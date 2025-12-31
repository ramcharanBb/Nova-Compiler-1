module {
  func.func @binary_ops(%arg0: tensor<1x8xi32>,%arg1: tensor<1x8xf32>,%arg11:tensor<4x8xf32>) -> tensor<f32> {

    %1 = nova.mae %arg0, %arg1 : tensor<1x8xi32>, tensor<1x8xf32>
    %2 = nova.mse %arg0, %arg1 : tensor<1x8xi32>, tensor<1x8xf32>
    %3 = nova.cce %arg0, %arg1 : tensor<1x8xi32>, tensor<1x8xf32>
    %4 = nova.bce %arg0, %arg1 : tensor<1x8xi32>, tensor<1x8xf32>
    return %1 : tensor<f32>
  }
}
