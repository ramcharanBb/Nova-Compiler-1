
func.func @test_reshape_fusion(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x8x4xf32>, %bias: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x4x4xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  
  // Matmul: [1, 4, 8] * [1, 8, 4] -> [1, 4, 4]
  // Dimensions: B=1, M=4, N=4, K=8
  // Maps: (b, m, n, k)
  // A: (b, m, k)
  // B: (b, k, n)
  // C: (b, m, n)
  %2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<1x4x8xf32>, tensor<1x8x4xf32>) outs(%1 : tensor<1x4x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.mulf %in, %in_0 : f32
    %5 = arith.addf %out, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x4x4xf32>

  // Expand Shape: [1, 4, 4] -> [1, 1, 4, 4]
  %3 = tensor.expand_shape %2 [[0, 1], [2], [3]] output_shape [1, 1, 4, 4] : tensor<1x4x4xf32> into tensor<1x1x4x4xf32>

  // Add Bias: [1, 1, 4, 4] + [1, 1, 4, 4]
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%3, %bias : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) outs(%3 : tensor<1x1x4x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.addf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x1x4x4xf32>

  return %result : tensor<1x1x4x4xf32>
}
