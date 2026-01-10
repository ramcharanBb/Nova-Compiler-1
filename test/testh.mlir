module {
  func.func @main1(%arg0: tensor<8x16xf32, #nova.device<"0">>, %arg1: tensor<16x64xf32, #nova.device<"0">>, %arg2: tensor<1x64xf32, #nova.device<"0">>, %arg3: tensor<64x64xf32, #nova.device<"0">>, %arg4: tensor<1x64xf32, #nova.device<"0">>, %arg5: tensor<64x32xf32, #nova.device<"0">>, %arg6: tensor<1x32xf32, #nova.device<"0">>, %arg7: tensor<32x32xf32, #nova.device<"0">>, %arg8: tensor<1x32xf32, #nova.device<"0">>, %arg9: tensor<32x10xf32, #nova.device<"0">>, %arg10: tensor<1x10xf32, #nova.device<"0">>) -> (tensor<f32, #nova.device<"0">>, tensor<16x64xf32, #nova.device<"0">>, tensor<1x64xf32, #nova.device<"0">>, tensor<64x64xf32, #nova.device<"0">>, tensor<1x64xf32, #nova.device<"0">>, tensor<64x32xf32, #nova.device<"0">>, tensor<1x32xf32, #nova.device<"0">>, tensor<32x32xf32, #nova.device<"0">>, tensor<1x32xf32, #nova.device<"0">>, tensor<32x10xf32, #nova.device<"0">>, tensor<1x10xf32, #nova.device<"0">>) attributes {llvm.emit_c_interface} {
    %0 = nova.matmul %arg0, %arg1 : tensor<8x16xf32, #nova.device<"0">>, tensor<16x64xf32, #nova.device<"0">>
    %1 = nova.add %0, %arg2 : tensor<8x64xf32, #nova.device<"0">>, tensor<1x64xf32, #nova.device<"0">>
    %2 = nova.gelu %1 : tensor<8x64xf32, #nova.device<"0">>
    %3 = nova.matmul %2, %arg3 : tensor<8x64xf32, #nova.device<"0">>, tensor<64x64xf32, #nova.device<"0">>
    %4 = nova.add %3, %arg4 : tensor<8x64xf32, #nova.device<"0">>, tensor<1x64xf32, #nova.device<"0">>
    %5 = nova.sigmoid %4 : tensor<8x64xf32, #nova.device<"0">>
    %6 = nova.matmul %5, %arg5 : tensor<8x64xf32, #nova.device<"0">>, tensor<64x32xf32, #nova.device<"0">>
    %7 = nova.add %6, %arg6 : tensor<8x32xf32, #nova.device<"0">>, tensor<1x32xf32, #nova.device<"0">>
    %8 = nova.tanh %7 : tensor<8x32xf32, #nova.device<"0">>
    %9 = nova.matmul %8, %arg7 : tensor<8x32xf32, #nova.device<"0">>, tensor<32x32xf32, #nova.device<"0">>
    %10 = nova.add %9, %arg8 : tensor<8x32xf32, #nova.device<"0">>, tensor<1x32xf32, #nova.device<"0">>
    %11 = nova.sigmoid %10 : tensor<8x32xf32, #nova.device<"0">>
    %12 = nova.matmul %11, %arg9 : tensor<8x32xf32, #nova.device<"0">>, tensor<32x10xf32, #nova.device<"0">>
    %13 = nova.add %12, %arg10 : tensor<8x10xf32, #nova.device<"0">>, tensor<1x10xf32, #nova.device<"0">>
    %14 = nova.constant {value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<8x10xf32, #nova.device<"0">>} : tensor<8x10xf32, #nova.device<"0">>
    %15 = nova.cce %13, %14 : tensor<8x10xf32, #nova.device<"0">>, tensor<8x10xf32, #nova.device<"0">>
    %16 = nova.constant {value = dense<1.000000e+00> : tensor<f32, #nova.device<"0">>} : tensor<f32, #nova.device<"0">>
    %17 = nova.constant {value = dense<0.000000e+00> : tensor<f32, #nova.device<"0">>} : tensor<f32, #nova.device<"0">>
    %18 = nova.add %16, %17 : tensor<f32, #nova.device<"0">>, tensor<f32, #nova.device<"0">>
    %19 = nova.constant {value = dense<9.99999971E-10> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %20 = nova.add %13, %19 : tensor<8x10xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %21 = nova.constant {value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<8x10xf32, #nova.device<"0">>} : tensor<8x10xf32, #nova.device<"0">>
    %22 = nova.div %21, %20 : tensor<8x10xf32, #nova.device<"0">>, tensor<8x10xf32, #nova.device<"0">>
    %23 = nova.constant {value = dense<-1.250000e-01> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %24 = nova.mul %22, %23 : tensor<8x10xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %25 = nova.mul %18, %24 : tensor<f32, #nova.device<"0">>, tensor<8x10xf32, #nova.device<"0">>
    %26 = nova.transpose %25 : tensor<8x10xf32, #nova.device<"0">>
    %27 = nova.reduce<sum> %26 dimension = [1] keepdims = true : tensor<10x8xf32, #nova.device<"0">>
    %28 = nova.transpose %27 : tensor<10x1xf32, #nova.device<"0">>
    %29 = nova.transpose %arg9 : tensor<32x10xf32, #nova.device<"0">>
    %30 = nova.matmul %25, %29 : tensor<8x10xf32, #nova.device<"0">>, tensor<10x32xf32, #nova.device<"0">>
    %31 = nova.transpose %11 : tensor<8x32xf32, #nova.device<"0">>
    %32 = nova.matmul %31, %25 : tensor<32x8xf32, #nova.device<"0">>, tensor<8x10xf32, #nova.device<"0">>
    %33 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %34 = nova.sub %33, %11 : tensor<1xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %35 = nova.mul %11, %34 : tensor<8x32xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %36 = nova.mul %30, %35 : tensor<8x32xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %37 = nova.transpose %36 : tensor<8x32xf32, #nova.device<"0">>
    %38 = nova.reduce<sum> %37 dimension = [1] keepdims = true : tensor<32x8xf32, #nova.device<"0">>
    %39 = nova.transpose %38 : tensor<32x1xf32, #nova.device<"0">>
    %40 = nova.transpose %arg7 : tensor<32x32xf32, #nova.device<"0">>
    %41 = nova.matmul %36, %40 : tensor<8x32xf32, #nova.device<"0">>, tensor<32x32xf32, #nova.device<"0">>
    %42 = nova.transpose %8 : tensor<8x32xf32, #nova.device<"0">>
    %43 = nova.matmul %42, %36 : tensor<32x8xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %44 = nova.mul %8, %8 : tensor<8x32xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %45 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %46 = nova.sub %45, %44 : tensor<1xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %47 = nova.mul %41, %46 : tensor<8x32xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %48 = nova.transpose %47 : tensor<8x32xf32, #nova.device<"0">>
    %49 = nova.reduce<sum> %48 dimension = [1] keepdims = true : tensor<32x8xf32, #nova.device<"0">>
    %50 = nova.transpose %49 : tensor<32x1xf32, #nova.device<"0">>
    %51 = nova.transpose %arg5 : tensor<64x32xf32, #nova.device<"0">>
    %52 = nova.matmul %47, %51 : tensor<8x32xf32, #nova.device<"0">>, tensor<32x64xf32, #nova.device<"0">>
    %53 = nova.transpose %5 : tensor<8x64xf32, #nova.device<"0">>
    %54 = nova.matmul %53, %47 : tensor<64x8xf32, #nova.device<"0">>, tensor<8x32xf32, #nova.device<"0">>
    %55 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %56 = nova.sub %55, %5 : tensor<1xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %57 = nova.mul %5, %56 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %58 = nova.mul %52, %57 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %59 = nova.transpose %58 : tensor<8x64xf32, #nova.device<"0">>
    %60 = nova.reduce<sum> %59 dimension = [1] keepdims = true : tensor<64x8xf32, #nova.device<"0">>
    %61 = nova.transpose %60 : tensor<64x1xf32, #nova.device<"0">>
    %62 = nova.transpose %arg3 : tensor<64x64xf32, #nova.device<"0">>
    %63 = nova.matmul %58, %62 : tensor<8x64xf32, #nova.device<"0">>, tensor<64x64xf32, #nova.device<"0">>
    %64 = nova.transpose %2 : tensor<8x64xf32, #nova.device<"0">>
    %65 = nova.matmul %64, %58 : tensor<64x8xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %66 = nova.mul %1, %1 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %67 = nova.mul %66, %1 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %68 = nova.constant {value = dense<4.471500e-02> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %69 = nova.mul %67, %68 : tensor<8x64xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %70 = nova.add %1, %69 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %71 = nova.constant {value = dense<0.797884583> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %72 = nova.mul %70, %71 : tensor<8x64xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %73 = nova.tanh %72 : tensor<8x64xf32, #nova.device<"0">>
    %74 = nova.constant {value = dense<0.134144992> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %75 = nova.mul %66, %74 : tensor<8x64xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %76 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %77 = nova.add %76, %75 : tensor<1xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %78 = nova.constant {value = dense<0.797884583> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %79 = nova.mul %77, %78 : tensor<8x64xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %80 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %81 = nova.add %80, %73 : tensor<1xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %82 = nova.constant {value = dense<5.000000e-01> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %83 = nova.mul %81, %82 : tensor<8x64xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %84 = nova.mul %73, %73 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %85 = nova.constant {value = dense<1.000000e+00> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %86 = nova.sub %85, %84 : tensor<1xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %87 = nova.mul %1, %86 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %88 = nova.mul %87, %79 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %89 = nova.constant {value = dense<5.000000e-01> : tensor<1xf32, #nova.device<"0">>} : tensor<1xf32, #nova.device<"0">>
    %90 = nova.mul %88, %89 : tensor<8x64xf32, #nova.device<"0">>, tensor<1xf32, #nova.device<"0">>
    %91 = nova.add %83, %90 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %92 = nova.mul %63, %91 : tensor<8x64xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    %93 = nova.transpose %92 : tensor<8x64xf32, #nova.device<"0">>
    %94 = nova.reduce<sum> %93 dimension = [1] keepdims = true : tensor<64x8xf32, #nova.device<"0">>
    %95 = nova.transpose %94 : tensor<64x1xf32, #nova.device<"0">>
    %96 = nova.transpose %arg0 : tensor<8x16xf32, #nova.device<"0">>
    %97 = nova.matmul %96, %92 : tensor<16x8xf32, #nova.device<"0">>, tensor<8x64xf32, #nova.device<"0">>
    return %15, %97, %95, %65, %61, %54, %50, %43, %39, %32, %28 : tensor<f32, #nova.device<"0">>, tensor<16x64xf32, #nova.device<"0">>, tensor<1x64xf32, #nova.device<"0">>, tensor<64x64xf32, #nova.device<"0">>, tensor<1x64xf32, #nova.device<"0">>, tensor<64x32xf32, #nova.device<"0">>, tensor<1x32xf32, #nova.device<"0">>, tensor<32x32xf32, #nova.device<"0">>, tensor<1x32xf32, #nova.device<"0">>, tensor<32x10xf32, #nova.device<"0">>, tensor<1x10xf32, #nova.device<"0">>
  }
}