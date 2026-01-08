module {
  func.func @binary_ops(%arg0: f64,%arg1: f64) -> tensor<8x8xf32, #nova.device<"1">> {
    %0=nova.constant {value = dense<1.0>  : tensor<8x8xf32,#nova.device<"1">>} : tensor<8x8xf32,#nova.device<"1">> 
    return %0 : tensor<8x8xf32, #nova.device<"1">>
  }
}
