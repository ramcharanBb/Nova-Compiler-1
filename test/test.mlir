module {
  func.func @main(%arg0: tensor<6x8xf32, #nova.device<"0">>) -> tensor<i1, #nova.device<"0">> attributes {llvm.emit_c_interface} {
    %0 = nova.reduce<any> %arg0 : tensor<6x8xf32, #nova.device<"0">>
    return %0 : tensor<i1, #nova.device<"0">>
  }
}