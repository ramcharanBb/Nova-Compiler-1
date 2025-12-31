#map_a = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
#map_b = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5, d4)>
#map_c = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>

module {
  func.func @matmul_5d(%A: tensor<2x4x8x128x128xf32>, 
                       %B: tensor<2x4x8x128x128xf32>, 
                       %C: tensor<2x4x8x128x128xf32>) -> tensor<2x4x8x128x128xf32> {
    
    %result = linalg.generic {
      indexing_maps = [#map_a, #map_b, #map_c],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<2x4x8x128x128xf32>, tensor<2x4x8x128x128xf32>)
      outs(%C : tensor<2x4x8x128x128xf32>) {
    
    ^bb0(%a_el: f32, %b_el: f32, %c_el: f32):
      %mul = arith.mulf %a_el, %b_el : f32
      %add = arith.addf %c_el, %mul : f32
      linalg.yield %add : f32
    } -> tensor<2x4x8x128x128xf32>
    
    return %result : tensor<2x4x8x128x128xf32>
  }
}