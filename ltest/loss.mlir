module {
  func.func @binary_ops(%arg0: f64,%arg1: f64,%arg2:i32) -> tensor<8x8xf64> {

    %1 = nova.rndm2d %arg0, %arg1,%arg2 : tensor<8x8xf64>
    return %1 : tensor<8x8xf64>
  }
}
