module {
  func.func @unary_ops(%arg0: tensor<1x8xf32, #nova.device<"1">>) -> tensor<1x8xf32, #nova.device<"1">> {

    // ARITHMETIC
    %0 = nova.square %arg0 : tensor<1x8xf32, #nova.device<"1">>
    %1 = nova.sqrt   %0    : tensor<1x8xf32, #nova.device<"1">>
    %2 = nova.neg    %1    : tensor<1x8xf32, #nova.device<"1">>
    %3 = nova.abs    %2    : tensor<1x8xf32, #nova.device<"1">>
    %4 = nova.sign   %3    : tensor<1x8xf32, #nova.device<"1">>
    %5 = nova.reciprocal %3 : tensor<1x8xf32, #nova.device<"1">>
     %51=nova.not %3:tensor<1x8xf32, #nova.device<"1">>
    // EXPONENTS
    %6 = nova.exp   %5 : tensor<1x8xf32, #nova.device<"1">>
    %7 = nova.exp2  %6 : tensor<1x8xf32, #nova.device<"1">>
    %8 = nova.log   %7 : tensor<1x8xf32, #nova.device<"1">>
    %9 = nova.log2  %8 : tensor<1x8xf32, #nova.device<"1">>
    %10 = nova.log10 %9 : tensor<1x8xf32, #nova.device<"1">>

    // TRIGONOMETRY
    %11 = nova.sin  %10 : tensor<1x8xf32, #nova.device<"1">>
    %12 = nova.cos  %11 : tensor<1x8xf32, #nova.device<"1">>
    %13 = nova.tan  %12 : tensor<1x8xf32, #nova.device<"1">>
    %14 = nova.asin %13 : tensor<1x8xf32, #nova.device<"1">>
    %15 = nova.acos %14 : tensor<1x8xf32, #nova.device<"1">>
    %16 = nova.atan %15 : tensor<1x8xf32, #nova.device<"1">>
    %17 = nova.sinh %16 : tensor<1x8xf32, #nova.device<"1">>
    %18 = nova.cosh %17 : tensor<1x8xf32, #nova.device<"1">>
    %19 = nova.tanh %18 : tensor<1x8xf32, #nova.device<"1">>
    %171 = nova.asinh %16 : tensor<1x8xf32, #nova.device<"1">>
    %181= nova.acosh %17 : tensor<1x8xf32, #nova.device<"1">>
    %191= nova.atanh %18 : tensor<1x8xf32, #nova.device<"1">>
    return %19 : tensor<1x8xf32, #nova.device<"1">>
  }
}
