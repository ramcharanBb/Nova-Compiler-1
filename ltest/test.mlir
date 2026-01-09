module {
  func.func @main1(%arg0: tensor<8x16xf32,#nova.device<"1">>, %arg1: tensor<16x32xf32,#nova.device<"1">>, %arg2: tensor<1x32xf32,#nova.device<"1">>, %arg3: tensor<32x10xf32,#nova.device<"1">>, %arg4: tensor<1x10xf32,#nova.device<"1">>) -> (tensor<f32,#nova.device<"1">>, tensor<16x32xf32,#nova.device<"1">>, tensor<1x32xf32,#nova.device<"1">>, tensor<32x10xf32,#nova.device<"1">>, tensor<1x10xf32,#nova.device<"1">>) attributes {llvm.emit_c_interface} {
    %0 = nova.matmul %arg0, %arg1 : tensor<8x16xf32,#nova.device<"1">>, tensor<16x32xf32,#nova.device<"1">>
    %1 = nova.add %0, %arg2 : tensor<8x32xf32,#nova.device<"1">>, tensor<1x32xf32,#nova.device<"1">>
    %2 = nova.relu %1 : tensor<8x32xf32,#nova.device<"1">>
    %3 = nova.matmul %2, %arg3 : tensor<8x32xf32,#nova.device<"1">>, tensor<32x10xf32,#nova.device<"1">>
    %4 = nova.add %3, %arg4 : tensor<8x10xf32,#nova.device<"1">>, tensor<1x10xf32,#nova.device<"1">>
    %5 = nova.constant {value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<8x10xf32,#nova.device<"1">>} : tensor<8x10xf32,#nova.device<"1">>
    %6 = nova.reduce<max> %4 dimension = [1] keepdims = true : tensor<8x10xf32,#nova.device<"1">>
    %7 = nova.sub %4, %6 : tensor<8x10xf32,#nova.device<"1">>, tensor<8x1xf32,#nova.device<"1">>
    %8 = nova.exp %7 : tensor<8x10xf32,#nova.device<"1">>
    %9 = nova.reduce<sum> %8 dimension = [1] keepdims = true : tensor<8x10xf32,#nova.device<"1">>
    %10 = nova.log %9 : tensor<8x1xf32,#nova.device<"1">>
    %11 = nova.sub %7, %10 : tensor<8x10xf32,#nova.device<"1">>, tensor<8x1xf32,#nova.device<"1">>
    %12 = nova.mul %5, %11 : tensor<8x10xf32,#nova.device<"1">>, tensor<8x10xf32,#nova.device<"1">>
    %13 = nova.reduce<sum> %12 dimension = [1] : tensor<8x10xf32,#nova.device<"1">>
    %14 = nova.constant {value = dense<-1.000000e+00> : tensor<8xf32,#nova.device<"1">>} : tensor<8xf32,#nova.device<"1">>
    %15 = nova.mul %13, %14 : tensor<8xf32,#nova.device<"1">>, tensor<8xf32,#nova.device<"1">>
    %16 = nova.reduce<mean> %15 : tensor<8xf32,#nova.device<"1">>
    %17 = nova.constant {value = dense<1.000000e+00> : tensor<f32,#nova.device<"1">>} : tensor<f32,#nova.device<"1">>
    %18 = nova.constant {value = dense<0.000000e+00> : tensor<f32,#nova.device<"1">>} : tensor<f32,#nova.device<"1">>
    %19 = nova.add %17, %18 : tensor<f32,#nova.device<"1">>, tensor<f32,#nova.device<"1">>
    %20 = nova.reduce<max> %4 dimension = [1] keepdims = true : tensor<8x10xf32,#nova.device<"1">>
    %21 = nova.sub %4, %20 : tensor<8x10xf32,#nova.device<"1">>, tensor<8x1xf32,#nova.device<"1">>
    %22 = nova.exp %21 : tensor<8x10xf32,#nova.device<"1">>
    %23 = nova.reduce<sum> %22 dimension = [1] keepdims = true : tensor<8x10xf32,#nova.device<"1">>
    %24 = nova.div %22, %23 : tensor<8x10xf32, #nova.device<"1">>, tensor<8x1xf32, #nova.device<"1">>
    %25 = nova.constant {value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<8x10xf32, #nova.device<"1">>} : tensor<8x10xf32, #nova.device<"1">>
    %26 = nova.sub %24, %25 : tensor<8x10xf32, #nova.device<"1">>, tensor<8x10xf32, #nova.device<"1">>
    %27 = nova.constant {value = dense<1.250000e-01> : tensor<1xf32, #nova.device<"1">>} : tensor<1xf32, #nova.device<"1">>
    %28 = nova.mul %26, %27 : tensor<8x10xf32, #nova.device<"1">>, tensor<1xf32, #nova.device<"1">>
    %29 = nova.mul %19, %28 : tensor<f32, #nova.device<"1">>, tensor<8x10xf32, #nova.device<"1">>
    %30 = nova.transpose %29 : tensor<8x10xf32, #nova.device<"1">>
    %31 = nova.reduce<sum> %30 dimension = [1] keepdims = true : tensor<10x8xf32, #nova.device<"1">>
    %32 = nova.transpose %31 : tensor<10x1xf32, #nova.device<"1">>
    %33 = nova.transpose %arg3 : tensor<32x10xf32, #nova.device<"1">>
    %34 = nova.matmul %29, %33 : tensor<8x10xf32, #nova.device<"1">>, tensor<10x32xf32, #nova.device<"1">>
    %35 = nova.transpose %2 : tensor<8x32xf32, #nova.device<"1">>
    %36 = nova.matmul %35, %29 : tensor<32x8xf32, #nova.device<"1">>, tensor<8x10xf32, #nova.device<"1">>
    %37 = nova.sign %1 : tensor<8x32xf32, #nova.device<"1">>
    %38 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"1">>} : tensor<1xf32, #nova.device<"1">>
    %39 = nova.add %38, %37 : tensor<1xf32, #nova.device<"1">>, tensor<8x32xf32, #nova.device<"1">>
    %40 = nova.constant {value = dense<5.000000e-01> : tensor<1xf32, #nova.device<"1">>} : tensor<1xf32, #nova.device<"1">>
    %41 = nova.mul %39, %40 : tensor<8x32xf32, #nova.device<"1">>, tensor<1xf32, #nova.device<"1">>
    %42 = nova.mul %34, %41 : tensor<8x32xf32, #nova.device<"1">>, tensor<8x32xf32, #nova.device<"1">>
    %43 = nova.transpose %42 : tensor<8x32xf32, #nova.device<"1">>
    %44 = nova.reduce<sum> %43 dimension = [1] keepdims = true : tensor<32x8xf32, #nova.device<"1">>
    %45 = nova.transpose %44 : tensor<32x1xf32, #nova.device<"1">>
    %46 = nova.transpose %arg0 : tensor<8x16xf32, #nova.device<"1">>
    %47 = nova.matmul %46, %42 : tensor<16x8xf32, #nova.device<"1">>, tensor<8x32xf32, #nova.device<"1">>
    return %16, %47, %45, %36, %32 : tensor<f32, #nova.device<"1">>, tensor<16x32xf32, #nova.device<"1">>, tensor<1x32xf32, #nova.device<"1">>, tensor<32x10xf32, #nova.device<"1">>, tensor<1x10xf32, #nova.device<"1">>
  }
}