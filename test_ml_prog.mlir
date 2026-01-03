module {
  ml_program.global private mutable @v(dense<0.0> : tensor<1024xf32>) : tensor<1024xf32>
  func.func @main1() -> tensor<1024xf32> {
    %0 = ml_program.global_load @v : tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
