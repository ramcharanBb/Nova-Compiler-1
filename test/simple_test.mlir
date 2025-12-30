module {
  // Helper Function: Embedding Lookup
  func.func @embedding_lookup(%indices: tensor<8x128xi32>, %W: tensor<1000x768xf32>) -> tensor<8x128x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c768 = arith.constant 768 : index
    
    // Initialize output tensor
    %init = tensor.empty() : tensor<8x128x768xf32>
    
    // Use linalg.generic for embedding lookup
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,  // indices
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>  // output
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%indices : tensor<8x128xi32>) outs(%init : tensor<8x128x768xf32>) {
    ^bb0(%idx: i32, %out: f32):
      %idx_cast = arith.index_cast %idx : i32 to index
      %d2 = linalg.index 2 : index
      %val = tensor.extract %W[%idx_cast, %d2] : tensor<1000x768xf32>
      linalg.yield %val : f32
    } -> tensor<8x128x768xf32>
    
    return %output : tensor<8x128x768xf32>
  }

}
