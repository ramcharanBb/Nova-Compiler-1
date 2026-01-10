// RUN: nova-opt %s --convert-nova-to-linalg | FileCheck %s

module {
  func.func @test_adam(%param: tensor<64xf32, #nova.device<"0">>, %m: tensor<64xf32, #nova.device<"0">>, %v: tensor<64xf32, #nova.device<"0">>, %grad: tensor<64xf32, #nova.device<"0">>) -> (tensor<64xf32, #nova.device<"0">>, tensor<64xf32, #nova.device<"0">>, tensor<64xf32, #nova.device<"0">>) {
    // CHECK: linalg.generic
    %new_param, %new_m, %new_v = nova.adam %param, %m, %v, %grad 
      {beta1 = 0.9 : f32, beta2 = 0.999 : f32, epsilon = 1.0e-8 : f32, lr = 0.001 : f32, t = 1 : i32} 
      : tensor<64xf32, #nova.device<"0">>, tensor<64xf32, #nova.device<"0">>, tensor<64xf32, #nova.device<"0">>, tensor<64xf32, #nova.device<"0">>
      
    return %new_param, %new_m, %new_v : tensor<64xf32, #nova.device<"0">>, tensor<64xf32, #nova.device<"0">>, tensor<64xf32, #nova.device<"0">>
  }
}
