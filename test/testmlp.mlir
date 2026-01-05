module {
  // === Persistent State (Weights and Biases) ===
  // Layer 1: 8 -> 16
  ml_program.global private mutable @W1(dense<0.0> : tensor<8x16xf32>) : tensor<8x16xf32>
  ml_program.global private mutable @b1(dense<0.0> : tensor<16xf32>) : tensor<16xf32>
  
  // Layer 2: 16 -> 4
  ml_program.global private mutable @W2(dense<0.0> : tensor<16x4xf32>) : tensor<16x4xf32>
  ml_program.global private mutable @b2(dense<0.0> : tensor<4xf32>) : tensor<4xf32>

  // === Main Entry Point ===

  func.func @main1() -> tensor<4x1xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index // 10 iterations
    
    // Learning rate using the new scalarconst op
    %lr = nova.scalarconst {value = 0.01 : f64} : tensor<1xf32>

    // 1. Initial Data Setup (Deterministic Fixed Input)
    %X = nova.constant {value = dense<[
      [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
      [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
      [1.0, 1.0, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0],
      [0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 1.1, 1.1]
    ]> : tensor<4x8xf32>} : tensor<4x8xf32>
    
    %labels = nova.constant {value = dense<[
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]> : tensor<4x4xf32>} : tensor<4x4xf32>

    // 2. Initial Weights Setup (Using updated rndm2d signature)
    %min = arith.constant -0.05 : f64
    %max = arith.constant 0.05 : f64
    
    %W1_i = nova.rndm2d %min, %max : tensor<8x16xf32>
    ml_program.global_store @W1 = %W1_i : tensor<8x16xf32>
    
    %W2_i = nova.rndm2d %min, %max : tensor<16x4xf32>
    ml_program.global_store @W2 = %W2_i : tensor<16x4xf32>

    // 3. Training Loop
    scf.for %iv = %c0 to %c10 step %c1 {
      
      // Load current parameters
      %W1_c = ml_program.global_load @W1 : tensor<8x16xf32>
      %b1_c = ml_program.global_load @b1 : tensor<16xf32>
      %W2_c = ml_program.global_load @W2 : tensor<16x4xf32>
      %b2_c = ml_program.global_load @b2 : tensor<4xf32>

      // 3.1. Forward Pass
      %z1_2d = nova.matmul %X, %W1_c : tensor<4x8xf32>, tensor<8x16xf32>
      %z1 = nova.add %z1_2d, %b1_c : tensor<4x16xf32>, tensor<16xf32>
      %a1 = nova.relu %z1 : tensor<4x16xf32>

      %z2_2d = nova.matmul %a1, %W2_c : tensor<4x16xf32>, tensor<16x4xf32>
      %preds = nova.add %z2_2d, %b2_c : tensor<4x4xf32>, tensor<4xf32>

      // 3.2. Backward Pass
      // delta2 = preds - labels (Batch size 4)
      %delta2 = nova.sub %preds, %labels : tensor<4x4xf32>, tensor<4x4xf32>
      
      // Update for Layer 2
      %a1_T = nova.transpose %a1 axes1=0 axes2=1 : tensor<4x16xf32>
      %dW2 = nova.matmul %a1_T, %delta2 : tensor<16x4xf32>, tensor<4x4xf32>
      %db2 = nova.reduce<sum> %delta2 dimension = [0] : tensor<4x4xf32>
      
      // Backprop to Hidden Layer (Layer 1)
      %W2_T = nova.transpose %W2_c axes1=0 axes2=1 : tensor<16x4xf32>
      %delta1_raw = nova.matmul %delta2, %W2_T : tensor<4x4xf32>, tensor<4x16xf32>
      
      // Update for Layer 1
      %X_T = nova.transpose %X axes1=0 axes2=1 : tensor<4x8xf32>
      %dW1 = nova.matmul %X_T, %delta1_raw : tensor<8x4xf32>, tensor<4x16xf32>
      %db1 = nova.reduce<sum> %delta1_raw dimension = [0] : tensor<4x16xf32>
      
      // 3.3. Batch Gradient Descent Updates
      %dW2_s = nova.mul %dW2, %lr : tensor<16x4xf32>, tensor<1xf32>
      %db2_s = nova.mul %db2, %lr : tensor<4xf32>, tensor<1xf32>
      %dW1_s = nova.mul %dW1, %lr : tensor<8x16xf32>, tensor<1xf32>
      %db1_s = nova.mul %db1, %lr : tensor<16xf32>, tensor<1xf32>
      
      %W2_n = nova.sub %W2_c, %dW2_s : tensor<16x4xf32>, tensor<16x4xf32>
      %b2_n = nova.sub %b2_c, %db2_s : tensor<4xf32>, tensor<4xf32>
      %W1_n = nova.sub %W1_c, %dW1_s : tensor<8x16xf32>, tensor<8x16xf32>
      %b1_n = nova.sub %b1_c, %db1_s : tensor<16xf32>, tensor<16xf32>
      
      ml_program.global_store @W2 = %W2_n : tensor<16x4xf32>
      ml_program.global_store @b2 = %b2_n : tensor<4xf32>4
      ml_program.global_store @W1 = %W1_n : tensor<8x16xf32>
      ml_program.global_store @b1 = %b1_n : tensor<16xf32>
    }
    
    %final_b2_seq = ml_program.global_load @b2 : tensor<4xf32>
    %final_b2 = tensor.expand_shape %final_b2_seq [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
    return %final_b2 : tensor<4x1xf32>
  }
}
