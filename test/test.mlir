module {
  func.func @main1(%arg0: tensor<8x8xf32,1>,%arg1: tensor<8x8xf32,1>) -> tensor<f32> {
    %4 = nova.bce %arg0, %arg1 : tensor<8x8xf32,1>, tensor<8x8xf32,1>
    return %4 : tensor<f32>
  }
}