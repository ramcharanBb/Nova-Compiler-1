module attributes {gpu.container_module} {
    func.func @large_3d_add(%arg0: memref<1028x1028x1028xf32>, %arg1: memref<1028x1028x1028xf32>, %arg2: memref<1028x1028x1028xf32>) {
        linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                affine_map<(d0, d1, d2) -> (d0, d1, d2)>
            ],
            iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%arg0, %arg1 : memref<1028x1028x1028xf32>, memref<1028x1028x1028xf32>)
          outs(%arg2 : memref<1028x1028x1028xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
            %res = arith.addf %in0, %in1 : f32
            linalg.yield %res : f32
        }
        return
    }
}
