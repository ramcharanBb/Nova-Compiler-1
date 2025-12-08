module {
  func.func @main(%arg0: tensor<2x2x3x3xf32>,%arg1:tensor<2x2x3x3xf32>,%arg3:tensor<2x2x3x3xf32> ) -> tensor<2x2x3x3xf32> {
  %0 = nova.add %arg0,%arg1 : tensor<2x2x3x3xf32>,tensor<2x2x3x3xf32>
  %1 =nova.matmul %arg0,%arg1 : tensor<2x2x3x3xf32>,tensor<2x2x3x3xf32>
  %2 = nova.add %arg3,%1 : tensor<2x2x3x3xf32>,tensor<2x2x3x3xf32>
  return %2 :tensor<2x2x3x3xf32>
  }}
//module {
 // func.func @main(%arg0: tensor<3x3xcomplex<f32>>,%arg1:tensor<2x2x3x3xf32>,%arg3:tensor<2x2x3x3xf32> ) -> tensor<3x3xcomplex<f32>> {
//  %0 = nova.add %arg0,%arg1 : tensor<3x3xcomplex<f32>>,tensor<2x2x3x3xf32>
//  return %0 :tensor<3x3xcomplex<f32>>
//  }
//}