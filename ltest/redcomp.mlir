module {
  func.func @compare_and_reduce(
    %arg0: tensor<4x8xf32, #nova.device<"1">>,
    %arg1: tensor<4x8xf32, #nova.device<"1">>
  ) -> (tensor<4x8xi1, #nova.device<"1">>, tensor<8xf32, #nova.device<"1">>, tensor<4xi32, #nova.device<"1">>, tensor<8xi32, #nova.device<"1">>) {

    // ============================================================
    // COMPARISON OPERATIONS
    // ============================================================

    %cmp_eq  = nova.compare<eq>  %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>

    %cmp_gt  = nova.compare<gt>  %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>

    %cmp_le  = nova.compare<le>  %arg0, %arg1
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>

    // ============================================================
    // REDUCTION OPERATIONS
    // ============================================================

    // Reduce along dimension 0 → shape: 8
    %sum_d0 = nova.reduce<sum> %arg0 dimension = [0]
      : tensor<4x8xf32, #nova.device<"1">>

    // Reduce along dimension 1, keep dims → shape: 4x1
    %max_d1_keep = nova.reduce<max> %arg0 dimension = [1] keepdims = true
      : tensor<4x8xf32, #nova.device<"1">>

    // Reduce all dimensions → scalar
    %mean_all = nova.reduce<mean> %arg0
      : tensor<4x8xf32, #nova.device<"1">>

    // ============================================================
    // ARGMAX
    // ============================================================

    // Argmax along dimension 1 → returns indices
    %argmax_d1 = nova.argmax %arg0 dimension = 1
      : tensor<4x8xf32, #nova.device<"1">>

    // ============================================================
    // ARGMIN
    // ============================================================

    // Argmin along dimension 0 → returns indices
    %argmin_d0 = nova.argmin %arg0 dimension = 0
      : tensor<4x8xf32, #nova.device<"1">>

    // ============================================================
    // RETURN (pick representative results)
    // ============================================================

    return %cmp_eq, %sum_d0, %argmax_d1, %argmin_d0
      : tensor<4x8xi1, #nova.device<"1">>, tensor<8xf32, #nova.device<"1">>, tensor<4xi32, #nova.device<"1">>, tensor<8xi32, #nova.device<"1">>
  }
}
//../build/tools/nova-opt/nova-opt redcomp.mlir -convert-nova-to-tosa -convert-nova-to-linalg --debug --canonicalize