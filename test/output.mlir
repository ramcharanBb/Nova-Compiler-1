module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(8 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(10 : index) : i64
    %8 = llvm.mlir.constant(16 : index) : i64
    %9 = llvm.getelementptr %2[80] : (!llvm.ptr) -> !llvm.ptr, f32
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.add %10, %1 : i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.sub %1, %6 : i64
    %15 = llvm.add %13, %14 : i64
    %16 = llvm.urem %15, %1 : i64
    %17 = llvm.sub %15, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    llvm.br ^bb1(%4 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb5
    %20 = llvm.icmp "slt" %19, %5 : i64
    llvm.cond_br %20, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%4 : i64)
  ^bb3(%21: i64):  // 2 preds: ^bb2, ^bb4
    %22 = llvm.icmp "slt" %21, %7 : i64
    llvm.cond_br %22, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %23 = llvm.mul %19, %7 overflow<nsw, nuw> : i64
    %24 = llvm.add %23, %21 overflow<nsw, nuw> : i64
    %25 = llvm.getelementptr inbounds|nuw %18[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %25 : f32, !llvm.ptr
    %26 = llvm.add %21, %6 : i64
    llvm.br ^bb3(%26 : i64)
  ^bb5:  // pred: ^bb3
    %27 = llvm.add %19, %6 : i64
    llvm.br ^bb1(%27 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%4 : i64)
  ^bb7(%28: i64):  // 2 preds: ^bb6, ^bb14
    %29 = llvm.icmp "slt" %28, %5 : i64
    llvm.cond_br %29, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%4 : i64)
  ^bb9(%30: i64):  // 2 preds: ^bb8, ^bb13
    %31 = llvm.icmp "slt" %30, %7 : i64
    llvm.cond_br %31, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%4 : i64)
  ^bb11(%32: i64):  // 2 preds: ^bb10, ^bb12
    %33 = llvm.icmp "slt" %32, %8 : i64
    llvm.cond_br %33, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %34 = llvm.mul %28, %8 overflow<nsw, nuw> : i64
    %35 = llvm.add %34, %32 overflow<nsw, nuw> : i64
    %36 = llvm.getelementptr inbounds|nuw %arg1[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = llvm.load %36 : !llvm.ptr -> f32
    %38 = llvm.mul %32, %7 overflow<nsw, nuw> : i64
    %39 = llvm.add %38, %30 overflow<nsw, nuw> : i64
    %40 = llvm.getelementptr inbounds|nuw %arg8[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.mul %28, %7 overflow<nsw, nuw> : i64
    %43 = llvm.add %42, %30 overflow<nsw, nuw> : i64
    %44 = llvm.getelementptr inbounds|nuw %18[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %45 = llvm.load %44 : !llvm.ptr -> f32
    %46 = llvm.fmul %37, %41 : f32
    %47 = llvm.fadd %46, %45 : f32
    %48 = llvm.mul %28, %7 overflow<nsw, nuw> : i64
    %49 = llvm.add %48, %30 overflow<nsw, nuw> : i64
    %50 = llvm.getelementptr inbounds|nuw %18[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %47, %50 : f32, !llvm.ptr
    %51 = llvm.add %32, %6 : i64
    llvm.br ^bb11(%51 : i64)
  ^bb13:  // pred: ^bb11
    %52 = llvm.add %30, %6 : i64
    llvm.br ^bb9(%52 : i64)
  ^bb14:  // pred: ^bb9
    %53 = llvm.add %28, %6 : i64
    llvm.br ^bb7(%53 : i64)
  ^bb15:  // pred: ^bb7
    %54 = llvm.getelementptr %2[80] : (!llvm.ptr) -> !llvm.ptr, f32
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.add %55, %1 : i64
    %57 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.sub %1, %6 : i64
    %60 = llvm.add %58, %59 : i64
    %61 = llvm.urem %60, %1 : i64
    %62 = llvm.sub %60, %61 : i64
    %63 = llvm.inttoptr %62 : i64 to !llvm.ptr
    llvm.br ^bb16(%4 : i64)
  ^bb16(%64: i64):  // 2 preds: ^bb15, ^bb20
    %65 = llvm.icmp "slt" %64, %5 : i64
    llvm.cond_br %65, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%4 : i64)
  ^bb18(%66: i64):  // 2 preds: ^bb17, ^bb19
    %67 = llvm.icmp "slt" %66, %7 : i64
    llvm.cond_br %67, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %68 = llvm.mul %4, %7 overflow<nsw, nuw> : i64
    %69 = llvm.add %68, %66 overflow<nsw, nuw> : i64
    %70 = llvm.getelementptr inbounds|nuw %arg15[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %71 = llvm.load %70 : !llvm.ptr -> f32
    %72 = llvm.mul %64, %7 overflow<nsw, nuw> : i64
    %73 = llvm.add %72, %66 overflow<nsw, nuw> : i64
    %74 = llvm.getelementptr inbounds|nuw %63[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %71, %74 : f32, !llvm.ptr
    %75 = llvm.add %66, %6 : i64
    llvm.br ^bb18(%75 : i64)
  ^bb20:  // pred: ^bb18
    %76 = llvm.add %64, %6 : i64
    llvm.br ^bb16(%76 : i64)
  ^bb21:  // pred: ^bb16
    %77 = llvm.getelementptr %2[80] : (!llvm.ptr) -> !llvm.ptr, f32
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.add %78, %1 : i64
    %80 = llvm.call @malloc(%79) : (i64) -> !llvm.ptr
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.sub %1, %6 : i64
    %83 = llvm.add %81, %82 : i64
    %84 = llvm.urem %83, %1 : i64
    %85 = llvm.sub %83, %84 : i64
    %86 = llvm.inttoptr %85 : i64 to !llvm.ptr
    llvm.br ^bb22(%4 : i64)
  ^bb22(%87: i64):  // 2 preds: ^bb21, ^bb26
    %88 = llvm.icmp "slt" %87, %5 : i64
    llvm.cond_br %88, ^bb23, ^bb27
  ^bb23:  // pred: ^bb22
    llvm.br ^bb24(%4 : i64)
  ^bb24(%89: i64):  // 2 preds: ^bb23, ^bb25
    %90 = llvm.icmp "slt" %89, %7 : i64
    llvm.cond_br %90, ^bb25, ^bb26
  ^bb25:  // pred: ^bb24
    %91 = llvm.mul %87, %7 overflow<nsw, nuw> : i64
    %92 = llvm.add %91, %89 overflow<nsw, nuw> : i64
    %93 = llvm.getelementptr inbounds|nuw %18[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %94 = llvm.load %93 : !llvm.ptr -> f32
    %95 = llvm.mul %87, %7 overflow<nsw, nuw> : i64
    %96 = llvm.add %95, %89 overflow<nsw, nuw> : i64
    %97 = llvm.getelementptr inbounds|nuw %63[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.load %97 : !llvm.ptr -> f32
    %99 = llvm.fadd %94, %98 : f32
    %100 = llvm.mul %87, %7 overflow<nsw, nuw> : i64
    %101 = llvm.add %100, %89 overflow<nsw, nuw> : i64
    %102 = llvm.getelementptr inbounds|nuw %86[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %99, %102 : f32, !llvm.ptr
    %103 = llvm.add %89, %6 : i64
    llvm.br ^bb24(%103 : i64)
  ^bb26:  // pred: ^bb24
    %104 = llvm.add %87, %6 : i64
    llvm.br ^bb22(%104 : i64)
  ^bb27:  // pred: ^bb22
    %105 = llvm.getelementptr %2[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %106 = llvm.ptrtoint %105 : !llvm.ptr to i64
    %107 = llvm.add %106, %1 : i64
    %108 = llvm.call @malloc(%107) : (i64) -> !llvm.ptr
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.sub %1, %6 : i64
    %111 = llvm.add %109, %110 : i64
    %112 = llvm.urem %111, %1 : i64
    %113 = llvm.sub %111, %112 : i64
    %114 = llvm.inttoptr %113 : i64 to !llvm.ptr
    llvm.br ^bb28(%4 : i64)
  ^bb28(%115: i64):  // 2 preds: ^bb27, ^bb29
    %116 = llvm.icmp "slt" %115, %5 : i64
    llvm.cond_br %116, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    %117 = llvm.getelementptr inbounds|nuw %114[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %117 : f32, !llvm.ptr
    %118 = llvm.add %115, %6 : i64
    llvm.br ^bb28(%118 : i64)
  ^bb30:  // pred: ^bb28
    llvm.br ^bb31(%4 : i64)
  ^bb31(%119: i64):  // 2 preds: ^bb30, ^bb35
    %120 = llvm.icmp "slt" %119, %5 : i64
    llvm.cond_br %120, ^bb32, ^bb36
  ^bb32:  // pred: ^bb31
    llvm.br ^bb33(%4 : i64)
  ^bb33(%121: i64):  // 2 preds: ^bb32, ^bb34
    %122 = llvm.icmp "slt" %121, %7 : i64
    llvm.cond_br %122, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %123 = llvm.mul %119, %7 overflow<nsw, nuw> : i64
    %124 = llvm.add %123, %121 overflow<nsw, nuw> : i64
    %125 = llvm.getelementptr inbounds|nuw %86[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.load %125 : !llvm.ptr -> f32
    %127 = llvm.getelementptr inbounds|nuw %114[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %128 = llvm.load %127 : !llvm.ptr -> f32
    %129 = llvm.fadd %126, %128 : f32
    %130 = llvm.getelementptr inbounds|nuw %114[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %129, %130 : f32, !llvm.ptr
    %131 = llvm.add %121, %6 : i64
    llvm.br ^bb33(%131 : i64)
  ^bb35:  // pred: ^bb33
    %132 = llvm.add %119, %6 : i64
    llvm.br ^bb31(%132 : i64)
  ^bb36:  // pred: ^bb31
    %133 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %134 = llvm.ptrtoint %133 : !llvm.ptr to i64
    %135 = llvm.add %134, %1 : i64
    %136 = llvm.call @malloc(%135) : (i64) -> !llvm.ptr
    %137 = llvm.ptrtoint %136 : !llvm.ptr to i64
    %138 = llvm.sub %1, %6 : i64
    %139 = llvm.add %137, %138 : i64
    %140 = llvm.urem %139, %1 : i64
    %141 = llvm.sub %139, %140 : i64
    %142 = llvm.inttoptr %141 : i64 to !llvm.ptr
    llvm.br ^bb37(%4 : i64)
  ^bb37(%143: i64):  // 2 preds: ^bb36, ^bb38
    %144 = llvm.icmp "slt" %143, %6 : i64
    llvm.cond_br %144, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %145 = llvm.getelementptr inbounds|nuw %142[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %145 : f32, !llvm.ptr
    %146 = llvm.add %143, %6 : i64
    llvm.br ^bb37(%146 : i64)
  ^bb39:  // pred: ^bb37
    llvm.br ^bb40(%4 : i64)
  ^bb40(%147: i64):  // 2 preds: ^bb39, ^bb44
    %148 = llvm.icmp "slt" %147, %5 : i64
    llvm.cond_br %148, ^bb41, ^bb45
  ^bb41:  // pred: ^bb40
    llvm.br ^bb42(%4 : i64)
  ^bb42(%149: i64):  // 2 preds: ^bb41, ^bb43
    %150 = llvm.icmp "slt" %149, %6 : i64
    llvm.cond_br %150, ^bb43, ^bb44
  ^bb43:  // pred: ^bb42
    %151 = llvm.add %147, %149 overflow<nsw, nuw> : i64
    %152 = llvm.getelementptr inbounds|nuw %114[%151] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %153 = llvm.load %152 : !llvm.ptr -> f32
    %154 = llvm.getelementptr inbounds|nuw %142[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %155 = llvm.load %154 : !llvm.ptr -> f32
    %156 = llvm.fadd %153, %155 : f32
    %157 = llvm.getelementptr inbounds|nuw %142[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %156, %157 : f32, !llvm.ptr
    %158 = llvm.add %149, %6 : i64
    llvm.br ^bb42(%158 : i64)
  ^bb44:  // pred: ^bb42
    %159 = llvm.add %147, %6 : i64
    llvm.br ^bb40(%159 : i64)
  ^bb45:  // pred: ^bb40
    %160 = llvm.insertvalue %136, %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %161 = llvm.insertvalue %142, %160[1] : !llvm.struct<(ptr, ptr, i64)> 
    %162 = llvm.insertvalue %4, %161[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @free(%12) : (!llvm.ptr) -> ()
    llvm.call @free(%57) : (!llvm.ptr) -> ()
    llvm.call @free(%80) : (!llvm.ptr) -> ()
    llvm.call @free(%108) : (!llvm.ptr) -> ()
    llvm.return %162 : !llvm.struct<(ptr, ptr, i64)>
  }
}

