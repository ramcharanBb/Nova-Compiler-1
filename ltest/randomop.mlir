module {
  func.func @binary_ops(%arg0: f64,%arg1: f64) -> tensor<8x8xf32> {
    %0 = nova.scalarconst {value = 1.0 : f64} : f64
    %1 = nova.rndm2d %0, %arg1 : tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }
}
