#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @main1() -> tensor<5x8xf32> {
    // Matrix A: 5x3
    %A = arith.constant dense<[[1.1, 1.2, 1.3],
                               [2.1, 2.2, 2.3],
                               [3.1, 3.2, 3.3],
                               [4.1, 4.2, 4.3],
                               [5.1, 5.2, 5.3]]> : tensor<5x3xf32>
    
    // Matrix B: 3x8
    %B = arith.constant dense<[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                               [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                               [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]]> : tensor<3x8xf32>
    
    // Matrix C: 5x8 (Initial values/Bias)
    %C = arith.constant dense<0.5> : tensor<5x8xf32>
    
    %result = linalg.generic {
      indexing_maps = [#map_a, #map_b, #map_c],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<5x3xf32>, tensor<3x8xf32>)
      outs(%C : tensor<5x8xf32>) {
    
    ^bb0(%a_el: f32, %b_el: f32, %c_el: f32):
      %mul = arith.mulf %a_el, %b_el : f32
      %add = arith.addf %c_el, %mul : f32
      linalg.yield %add : f32
    } -> tensor<5x8xf32>
    
    return %result : tensor<5x8xf32>
  }
}