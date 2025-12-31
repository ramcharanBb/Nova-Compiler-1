module {
  func.func @unary_ops(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {

    // ARITHMETIC
    %0 = nova.square %arg0 : tensor<1x8xf32>
    %1 = nova.sqrt   %0    : tensor<1x8xf32>
    %2 = nova.neg    %1    : tensor<1x8xf32>
    %3 = nova.abs    %2    : tensor<1x8xf32>
    %4 = nova.sign   %3    : tensor<1x8xf32>
    %5 = nova.reciprocal %3 : tensor<1x8xf32>
     %51=nova.not %3:tensor<1x8xf32>
    // EXPONENTS
    %6 = nova.exp   %5 : tensor<1x8xf32>
    %7 = nova.exp2  %6 : tensor<1x8xf32>
    %8 = nova.log   %7 : tensor<1x8xf32>
    %9 = nova.log2  %8 : tensor<1x8xf32>
    %10 = nova.log10 %9 : tensor<1x8xf32>

    // TRIGONOMETRY
    %11 = nova.sin  %10 : tensor<1x8xf32>
    %12 = nova.cos  %11 : tensor<1x8xf32>
    %13 = nova.tan  %12 : tensor<1x8xf32>
    %14 = nova.asin %13 : tensor<1x8xf32>
    %15 = nova.acos %14 : tensor<1x8xf32>
    %16 = nova.atan %15 : tensor<1x8xf32>
    %17 = nova.sinh %16 : tensor<1x8xf32>
    %18 = nova.cosh %17 : tensor<1x8xf32>
    %19 = nova.tanh %18 : tensor<1x8xf32>
    %171 = nova.asinh %16 : tensor<1x8xf32>
    %181= nova.acosh %17 : tensor<1x8xf32>
    %191= nova.atanh %18 : tensor<1x8xf32>
    return %19 : tensor<1x8xf32>
  }
}
