module {
  func.func @binary_ops(
    %arg0: tensor<1x8xf32>,
    %arg1: tensor<1x8xf32>
  ) -> (tensor<1x8xi32>){ 
%1=nova.empty_tensor () :tensor<1x8xi32>
return %1:tensor<1x8xi32>
}}