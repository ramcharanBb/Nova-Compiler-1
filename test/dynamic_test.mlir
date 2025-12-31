module attributes {gpu.container_module} {
    func.func @dynamic_add(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
        linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1) -> (d0, d1)>,
                affine_map<(d0, d1) -> (d0, d1)>,
                affine_map<(d0, d1) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel"]
        } ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
          outs(%arg2 : memref<?x?xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
            %res = arith.addf %in0, %in1 : f32
            linalg.yield %res : f32
        }
        return
    }
}
