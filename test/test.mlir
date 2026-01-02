module{
   func.func @main1() -> tensor<4x3xf32> {
    %arg0 = arith.constant dense<[[1.1, 2.2, 3.3], 
                                  [4.4, 5.5, 6.6],
                                  [13.0, 14.0, 15.0],
                                  [7.7, 8.8, 9.9]]> : tensor<4x3xf32>  
    %arg1 = arith.constant dense<[[0.1, 0.2, 0.3], 
                                  [0.4, 0.5, 0.6], 
                                  [4.4, 5.5, 6.6]]> : tensor<3x3xf32>
    %arg2 = arith.constant dense<[[10.0, 11.0, 12.0], 
                                  [0.1, 0.2, 0.3],
                                  [13.0, 14.0, 15.0],
                                  [16.0, 17.0, 18.0]]> : tensor<4x3xf32>
  %1 =nova.matmul %arg0,%arg1 : tensor<4x3xf32>,tensor<3x3xf32>
  %2 = nova.add %1,%arg2 : tensor<4x3xf32>, tensor<4x3xf32>
  return %2 :tensor<4x3xf32>
  }
}
  
