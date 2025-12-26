module {
  func.func @activation_ops(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {

    // ACTIVATION FUNCTIONS
    %0 = nova.relu    %arg0 : tensor<1x8xf32>
    %1 = nova.sigmoid %0    : tensor<1x8xf32>
    %2 = nova.softmax %1    : tensor<1x8xf32>
    %3 = nova.gelu    %2    : tensor<1x8xf32>

    return %3 : tensor<1x8xf32>
  }
}
//../build/tools/nova-opt/nova-opt activefunc.mlir --canonicalize --convert-nova-to-tosa --convert-nova-to-linalg 
 //../build/tools/nova-opt/nova-opt binary.mlir   --pass-pipeline='builtin.module(canonicalize,convert-nova-to-tosa,func.func(convert-nova-to-linalg,tosa-to-linalg-named,tosa-to-linalg))'
