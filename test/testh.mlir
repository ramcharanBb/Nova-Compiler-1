module {
 func.func @main(%arg0: tensor<3x3xcomplex<f32>>,%arg1:tensor<3x3xf32>,%arg3:tensor<1x3xi32> ) -> tensor<3x3xcomplex<f32>> {
  %0 = nova.add %arg0,%arg1 : tensor<3x3xcomplex<f32>>,tensor<3x3xf32>
  return %0 :tensor<3x3xcomplex<f32>>
  }
}
// trait working