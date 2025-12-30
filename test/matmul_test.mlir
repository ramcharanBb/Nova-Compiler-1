module {
  // Helper Function: Matrix Multiplication with nova
  func.func @matmul(%A: tensor<8x128x768xf32>, %B: tensor<768x768xf32>) -> tensor<8x128x768xf32> {
    // Reshape for matmul: collapse batch dimension
    %A_2d = tensor.collapse_shape %A [[0, 1], [2]] : tensor<8x128x768xf32> into tensor<1024x768xf32>
    
    // nova.matmul doesn't use explicit output buffer in its syntax based on NovaOps.td
    %result_2d = nova.matmul %A_2d, %B : tensor<1024x768xf32>, tensor<768x768xf32>
    
    %result = tensor.expand_shape %result_2d [[0, 1], [2]] output_shape [8, 128, 768] : tensor<1024x768xf32> into tensor<8x128x768xf32>
    return %result : tensor<8x128x768xf32>
  }
}
