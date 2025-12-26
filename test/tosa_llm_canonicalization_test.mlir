// RUN: mlir-opt --split-input-file -canonicalize %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Element-wise Arithmetic Optimizations (Add, Sub, Mul, Div, Reciprocal)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: @add_bcast_zero_int
func.func @add_bcast_zero_int(%arg0: tensor<4x2x3xi32>) -> tensor<4x2x3xi32> {
  // CHECK-NOT: tosa.add
  // CHECK: return %arg0
  %zeros = "tosa.const"() {values = dense<0> : tensor<1x1x1xi32>} : () -> tensor<1x1x1xi32>
  %1 = tosa.add %arg0, %zeros : (tensor<4x2x3xi32>, tensor<1x1x1xi32>) -> tensor<4x2x3xi32>
  return %1 : tensor<4x2x3xi32>
}

// -----

// CHECK-LABEL: @add_zero_int
func.func @add_zero_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.add
  %zeros = "tosa.const"() {values = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = tosa.add %arg0, %zeros : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @mul_one_float
func.func @mul_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %ones = "tosa.const"() {values = dense<1.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = tosa.mul %arg0, %ones, %shift : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @mul_zero_broadcast
func.func @mul_zero_broadcast(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2x3xf32>}
  // CHECK-NOT: tosa.mul
  %zeros = "tosa.const"() {values = dense<0.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %arg0, %zeros, %shift : (tensor<2x3xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<2x3xf32>

  // CHECK-NOT: tosa.mul
  // CHECK: return %[[ZERO]], %[[ZERO]]
  %2 = tosa.mul %zeros, %arg0, %shift : (tensor<1x1xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @fold_div_one_rhs_i32
func.func @fold_div_one_rhs_i32(%arg0: tensor<i32>) -> tensor<i32> {
  %one = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
  %div = tosa.intdiv %arg0, %one : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %arg0
  return %div : tensor<i32>
}

// -----

// CHECK-LABEL: @reciprocal_fold_splat
func.func @reciprocal_fold_splat() -> tensor<12x7xf32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}2.5{{0*}}e-01{{.*}}tensor<12x7xf32>
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {values = dense<4.0> : tensor<12x7xf32>} : () -> tensor<12x7xf32>
  %1 = "tosa.reciprocal"(%0) : (tensor<12x7xf32>) -> tensor<12x7xf32>
  return %1 : tensor<12x7xf32>
}

// ===----------------------------------------------------------------------===//
// Shape and View Optimizations (Reshape, Transpose, Slice, Concat)
// ===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @reshape_canonicalize
func.func @reshape_canonicalize(%arg0: tensor<?x10xf32>) -> tensor<?x10xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const_shape"() {values = dense<[-1, 10]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %arg0, %0 : (tensor<?x10xf32>, !tosa.shape<2>) -> tensor<?x10xf32>
  return %1 : tensor<?x10xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_double
func.func @reshape_canonicalize_double(%arg0: tensor<?x10xf32>) -> tensor<?x5xf32> {
  // CHECK: %[[VAL_0:.*]] = tosa.const_shape {values = dense<[-1, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[VAL_0]]
  // CHECK: return %[[VAL_1]]
  %cst0 = "tosa.const_shape"() <{values = dense<[5, -1]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %0 = tosa.reshape %arg0, %cst0 : (tensor<?x10xf32>, !tosa.shape<2>) -> tensor<5x?xf32>
  %cst1 = "tosa.const_shape"() <{values = dense<[-1, 5]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %cst1 : (tensor<5x?xf32>, !tosa.shape<2>) -> tensor<?x5xf32>
  return %1 : tensor<?x5xf32>
}

// -----

// CHECK-LABEL: @transpose_fold
func.func @transpose_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %1 = tosa.transpose %arg0 { perms = array<i32: 0, 1> }: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_splat
func.func @transpose_fold_splat() -> tensor<3x2xf32> {
  %input = "tosa.const"() {values = dense<4.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<4.000000e+00> : tensor<3x2xf32>
  %1 = tosa.transpose %input { perms = array<i32: 1, 0> }: (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @test_cancel_transpose_transpose
func.func @test_cancel_transpose_transpose(%arg0: tensor<1x2x3xi32>) -> (tensor<1x2x3xi32>) {
  // CHECK: return %arg0
  %1 = tosa.transpose %arg0 { perms = array<i32: 1, 2, 0> }: (tensor<1x2x3xi32>) -> tensor<2x3x1xi32>
  %3 = tosa.transpose %1 { perms = array<i32: 2, 0, 1> }: (tensor<2x3x1xi32>) -> tensor<1x2x3xi32>
  return %3 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: @test_prefer_compose_transpose
func.func @test_prefer_compose_transpose(%arg0: tensor<1x2x3x4xi32>) -> (tensor<4x3x2x1xi32>) {
  // CHECK: %[[VAL_2:.*]] = tosa.transpose %arg0 {perms = array<i32: 3, 2, 1, 0>} : (tensor<1x2x3x4xi32>) -> tensor<4x3x2x1xi32>
  // CHECK: return %[[VAL_2]]
  %1 = tosa.transpose %arg0 { perms = array<i32: 1, 2, 0, 3> }: (tensor<1x2x3x4xi32>) -> tensor<2x3x1x4xi32>
  %3 = tosa.transpose %1 { perms = array<i32: 3, 1, 0, 2> }: (tensor<2x3x1x4xi32>) -> tensor<4x3x2x1xi32>
  return %3 : tensor<4x3x2x1xi32>
}

// -----

// CHECK-LABEL: @concat_fold
func.func @concat_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.concat %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @fold_concats
func.func @fold_concats(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32> {
  // CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<1x1x7x7xf32>
  // CHECK: %[[VAL_2:.*]] = tosa.concat %[[VAL_1]], %[[VAL_0]], %[[VAL_0]], %[[VAL_1]] {axis = 1 : i32}
  %tmp = tensor.empty() : tensor<1x1x7x7xf32>
  %0 = tosa.concat %arg0, %arg0 {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x2x7x7xf32>
  %1 = tosa.concat %tmp, %0, %tmp {axis = 1 : i32} : (tensor<1x1x7x7xf32>, tensor<1x2x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x4x7x7xf32>
  return %1 : tensor<1x4x7x7xf32>
}

// -----

// CHECK-LABEL: @slice_splat
func.func @slice_splat() -> tensor<1x1x1xi32> {
  // CHECK: %[[SLICE:.+]] = "tosa.const"() <{values = dense<42> : tensor<1x1x1xi32>}
  %splat = "tosa.const"() {values = dense<42> : tensor<4x5x6xi32>} : () -> tensor<4x5x6xi32>
  %start = tosa.const_shape {values = dense<[1, 2, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %size = tosa.const_shape {values = dense<[1, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %slice= tosa.slice %splat, %start, %size : (tensor<4x5x6xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x1x1xi32>

  // CHECK: return %[[SLICE]]
  return %slice : tensor<1x1x1xi32>
}

// ===----------------------------------------------------------------------===//
// Comparison and Control Flow (Select, Equal, Greater, Clamp)
// ===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @select_same_value
func.func @select_same_value(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = tosa.select %arg0, %arg1, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @select_true_value
func.func @select_true_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c1 = "tosa.const"() {values = dense<1> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = tosa.select %c1, %arg0, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg0
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @clamp_f32_is_noop
func.func @clamp_f32_is_noop(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  // 0xFF800000 and 0x7F800000 are respectively negative and positive F32 infinity.
  %0 = tosa.clamp %arg0 {min_val = 0xFF800000 : f32, max_val = 0x7F800000 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @clamp_twice_is_single_clamp
func.func @clamp_twice_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.clamp %arg0 {max_val = 2 : i8, min_val = -2 : i8}
  %0 = tosa.clamp %arg0 {max_val = 4 : i8, min_val = -2 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_val = 2 : i8, min_val = -4 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @fold_eq_i32
func.func @fold_eq_i32(%arg0 : tensor<10xi32>) -> (tensor<10xi1>) {
  // CHECK: %[[TRUE:.+]] = "tosa.const"() <{values = dense<true> : tensor<10xi1>}
  %0 = tosa.equal %arg0, %arg0 : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  // CHECK: return %[[TRUE]]
  return %0 : tensor<10xi1>
}

// ===----------------------------------------------------------------------===//
// Reductions (ReduceSum, ReduceMax, etc.)
// ===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @reduce_all_fold
func.func @reduce_all_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_all %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_sum_fold
func.func @reduce_sum_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_sum %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @argmax_nofold
func.func @argmax_nofold(%arg0: tensor<?x1xf32>) -> tensor<1xi32> {
  // CHECK: tosa.argmax
  %0 = tosa.argmax %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// ===----------------------------------------------------------------------===//
// Data Type and Constants (Cast, Const)
// ===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @cast_fold
func.func @cast_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.cast %arg0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK: func.func @cast_float_to_float
func.func @cast_float_to_float() -> tensor<f16> {
  %splat = "tosa.const"() {values = dense<42.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SPLAT:.+]] = "tosa.const"() <{values = dense<4.200000e+01> : tensor<f16>}
  %cast = tosa.cast %splat : (tensor<f32>) -> tensor<f16>
  // CHECK: return %[[SPLAT]]
  return %cast : tensor<f16>
}
