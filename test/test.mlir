module {
  func.func @main(%arg0: tensor<1x3x3xf32>,%arg1:tensor<1x3x3xf32>,%arg2:tensor<1x3x3xf32> ) -> tensor<1x3x3xf32> {
  %1 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x3x3xf32>, tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  //%2 = nova.add %1,%arg2 : tensor<1x3x3xf32>,tensor<1x3x3xf32>
  return %1 :tensor<1x3x3xf32>
  }
  
//  func.func @main1(%arg0: tensor<1x3x3xf32>,%arg1:tensor<1x3x3xf32>,%arg2:tensor<3x3xi32> ) -> tensor<1x3x3xf32> {
//  %1 =nova.matmul %arg0,%arg1 : tensor<1x3x3xf32>,tensor<1x3x3xf32>
//  %2 = nova.add %1,%arg2 : tensor<1x3x3xf32>,tensor<3x3xi32>
//  return %2 :tensor<1x3x3xf32>
//  }

 // func.func @main2(%arg0: tensor<1x3x3xf32>,%arg1:tensor<1x3x3xf32>,%arg2:tensor<3x3xi16> ) -> tensor<1x3x3xf32> {
 // %1 =nova.matmul %arg0,%arg1 : tensor<1x3x3xf32>,tensor<1x3x3xf32>
 // %2 = nova.add %1,%arg2 : tensor<1x3x3xf32>,tensor<3x3xi16>
 // return %2 :tensor<1x3x3xf32>
  //}
  
 // func.func @main(%arg0: tensor<1x3x3xf32>,%arg1:tensor<1x3x3xf32>,%arg2:tensor<3x3xi64> ) -> tensor<3x3xf64> {
 // %1 =nova.matmul %arg0,%arg1 : tensor<1x3x3xf32>,tensor<1x3x3xf32>
 // %2 = nova.add %1,%arg2 : tensor<1x3x3xf32>,tensor<3x3xi64>
 // return %2 :tensor<3x3xf64>
 // }
  }
