module {
 func.func @main(%arg0: tensor<3x3xi32>,%arg1:tensor<3x3xf32>,%arg3:tensor<1x3xi32> ) -> tensor<3x3xf32> {
  %0 = nova.max %arg0,%arg1: tensor<3x3xi32>,tensor<3x3xf32>
  return %0 :tensor<3x3xf32>
  }
}

////////////
~/…/mlir-compiler $ build/tools/nova-opt/nova-opt test/test_fusion.mlir --linalg-generalize-named-ops --fuse-matmul-bias --one-shot-bufferize="bufferize-function-boundaries"

=== Running FuseMatmulBias Pass ===
Found fusible matmul + add pattern (with 0 reshapes)!
Successfully fused matmul + bias into single operation (thru reshapes)!
=== FuseMatmulBias Pass Complete ===
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_with_bias(%arg0: memref<4096x4096xf32, strided<[?, ?], offset: ?>>, %arg1: memref<4096x4096xf32, strided<[?, ?], offset: ?>>, %arg2: memref<4096x4096xf32, strided<[?, ?], offset: ?>>) -> memref<4096x4096xf32, strided<[?, ?], offset: ?>> {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4096x4096xf32, strided<[?, ?], offset: ?>>, memref<4096x4096xf32, strided<[?, ?], offset: ?>>) outs(%arg2 : memref<4096x4096xf32, strided<[?, ?], offset: ?>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return %arg2 : memref<4096x4096xf32, strided<[?, ?], offset: ?>>
  }
}