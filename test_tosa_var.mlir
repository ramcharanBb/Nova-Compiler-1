module {
  tosa.variable @v dense<0.0> : tensor<768x768xf32>
  func.func @test() {
    %0 = tosa.variable_read @v : tensor<768x768xf32>
    return
  }
}
