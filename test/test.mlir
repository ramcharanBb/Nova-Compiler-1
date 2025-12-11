module {
  func.func @main(%arg0: tensor<3x3xf32>,%arg1:tensor<3x3xf32>,%arg2:tensor<3x3xf16> ) -> tensor<3x3xf32> {
  %1 =nova.matmul %arg0,%arg1 : tensor<3x3xf32>,tensor<3x3xf32>
  %2 = nova.add %1,%arg2 : tensor<3x3xf32>,tensor<3x3xf16>
  return %2 :tensor<3x3xf32>
  }}
