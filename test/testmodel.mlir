module {
  // === Global Persistent State (Weights and Biases) ===
  ml_program.global private mutable @WQ(dense<0.0> : tensor<768x768xf32>) : tensor<768x768xf32>
  ml_program.global private mutable @WK(dense<0.0> : tensor<768x768xf32>) : tensor<768x768xf32>
  ml_program.global private mutable @WV(dense<0.0> : tensor<768x768xf32>) : tensor<768x768xf32>
  ml_program.global private mutable @WO(dense<0.0> : tensor<768x768xf32>) : tensor<768x768xf32>
  ml_program.global private mutable @W1(dense<0.0> : tensor<768x768xf32>) : tensor<768x768xf32>
  ml_program.global private mutable @W2(dense<0.0> : tensor<768x768xf32>) : tensor<768x768xf32>

  ml_program.global private mutable @bQ(dense<0.0> : tensor<768xf32>) : tensor<768xf32>
  ml_program.global private mutable @bK(dense<0.0> : tensor<768xf32>) : tensor<768xf32>
  ml_program.global private mutable @bV(dense<0.0> : tensor<768xf32>) : tensor<768xf32>
  ml_program.global private mutable @bO(dense<0.0> : tensor<768xf32>) : tensor<768xf32>
  ml_program.global private mutable @b1(dense<0.0> : tensor<768xf32>) : tensor<768xf32>
  ml_program.global private mutable @b2(dense<0.0> : tensor<768xf32>) : tensor<768xf32>

  // === Main Entry Point ===

  func.func @main1() -> tensor<768x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    
    %min = arith.constant -0.1 : f64
    %max = arith.constant 0.1 : f64
    %seed = arith.constant 42 : i32

    %scale = nova.constant {value = dense<0.036> : tensor<1xf32>} : tensor<1xf32>
    %lr = nova.constant {value = dense<0.01> : tensor<1xf32>} : tensor<1xf32>

    // 1. Initial Weights Setup
    %WQ_0 = nova.rndm2d %min, %max, %seed : tensor<768x768xf32>
    ml_program.global_store @WQ = %WQ_0 : tensor<768x768xf32>
    
    %WK_0 = nova.rndm2d %min, %max, %seed : tensor<768x768xf32>
    ml_program.global_store @WK = %WK_0 : tensor<768x768xf32>
    
    %WV_0 = nova.rndm2d %min, %max, %seed : tensor<768x768xf32>
    ml_program.global_store @WV = %WV_0 : tensor<768x768xf32>
    
    %WO_0 = nova.rndm2d %min, %max, %seed : tensor<768x768xf32>
    ml_program.global_store @WO = %WO_0 : tensor<768x768xf32>
    
    %W1_0 = nova.rndm2d %min, %max, %seed : tensor<768x768xf32>
    ml_program.global_store @W1 = %W1_0 : tensor<768x768xf32>
    
    %W2_0 = nova.rndm2d %min, %max, %seed : tensor<768x768xf32>
    ml_program.global_store @W2 = %W2_0 : tensor<768x768xf32>

    // 2. Training Loop
    scf.for %iv = %c0 to %c5 step %c1 {
      
      // 2.1. Batch Preparation (Batch 1, Seq 128, Hidden 768)
      %X = nova.rndm2d %min, %max, %seed : tensor<128x768xf32>
      %labels = nova.rndm2d %min, %max, %seed : tensor<128x768xf32>
      
      // Load current parameters
      %WQ_cur = ml_program.global_load @WQ : tensor<768x768xf32>
      %WK_cur = ml_program.global_load @WK : tensor<768x768xf32>
      %WV_cur = ml_program.global_load @WV : tensor<768x768xf32>
      %WO_cur = ml_program.global_load @WO : tensor<768x768xf32>
      %W1_cur = ml_program.global_load @W1 : tensor<768x768xf32>
      %W2_cur = ml_program.global_load @W2 : tensor<768x768xf32>

      %bQ_cur = ml_program.global_load @bQ : tensor<768xf32>
      %bK_cur = ml_program.global_load @bK : tensor<768xf32>
      %bV_cur = ml_program.global_load @bV : tensor<768xf32>
      %bO_cur = ml_program.global_load @bO : tensor<768xf32>
      %b1_cur = ml_program.global_load @b1 : tensor<768xf32>
      %b2_cur = ml_program.global_load @b2 : tensor<768xf32>

      // 2.2. Forward Pass
      %res_Q_2d = nova.matmul %X, %WQ_cur : tensor<128x768xf32>, tensor<768x768xf32>
      %Q = nova.add %res_Q_2d, %bQ_cur : tensor<128x768xf32>, tensor<768xf32>

      %res_K_2d = nova.matmul %X, %WK_cur : tensor<128x768xf32>, tensor<768x768xf32>
      %K = nova.add %res_K_2d, %bK_cur : tensor<128x768xf32>, tensor<768xf32>

      %res_V_2d = nova.matmul %X, %WV_cur : tensor<128x768xf32>, tensor<768x768xf32>
      %V = nova.add %res_V_2d, %bV_cur : tensor<128x768xf32>, tensor<768xf32>

      %KT = nova.transpose %K axes1=0 axes2=1 : tensor<128x768xf32>
      %scores_raw = nova.matmul %Q, %KT : tensor<128x768xf32>, tensor<768x128xf32>
      %scores_scaled = nova.mul %scores_raw, %scale : tensor<128x128xf32>, tensor<1xf32>
      %probs = nova.softmax %scores_scaled dimension=1 : tensor<128x128xf32>
      
      %context = nova.matmul %probs, %V : tensor<128x128xf32>, tensor<128x768xf32>

      %res_O_2d = nova.matmul %context, %WO_cur : tensor<128x768xf32>, tensor<768x768xf32>
      %attn_out = nova.add %res_O_2d, %bO_cur : tensor<128x768xf32>, tensor<768xf32>

      %x_attn = nova.add %X, %attn_out : tensor<128x768xf32>, tensor<128x768xf32>

      %res_FF1_2d = nova.matmul %x_attn, %W1_cur : tensor<128x768xf32>, tensor<768x768xf32>
      %FF1 = nova.add %res_FF1_2d, %b1_cur : tensor<128x768xf32>, tensor<768xf32>
      %FF1_act = nova.relu %FF1 : tensor<128x768xf32>

      %res_FF2_2d = nova.matmul %FF1_act, %W2_cur : tensor<128x768xf32>, tensor<768x768xf32>
      %preds = nova.add %res_FF2_2d, %b2_cur : tensor<128x768xf32>, tensor<768xf32>

      // 2.3. Backward Pass
      %diff = nova.sub %preds, %labels : tensor<128x768xf32>, tensor<128x768xf32>
      %X_T = nova.transpose %X axes1=0 axes2=1 : tensor<128x768xf32>
      
      %dW_common = nova.matmul %X_T, %diff : tensor<768x128xf32>, tensor<128x768xf32>
      %db_common = nova.reduce<sum> %diff dimension = [0] : tensor<128x768xf32>
      
      // 2.4. Batch Update
      %grad_scaled = nova.mul %dW_common, %lr : tensor<768x768xf32>, tensor<1xf32>
      %db_scaled = nova.mul %db_common, %lr : tensor<768xf32>, tensor<1xf32>

      %WQ_next = nova.sub %WQ_cur, %grad_scaled : tensor<768x768xf32>, tensor<768x768xf32>
      ml_program.global_store @WQ = %WQ_next : tensor<768x768xf32>
      
      %WK_next = nova.sub %WK_cur, %grad_scaled : tensor<768x768xf32>, tensor<768x768xf32>
      ml_program.global_store @WK = %WK_next : tensor<768x768xf32>

      %WV_next = nova.sub %WV_cur, %grad_scaled : tensor<768x768xf32>, tensor<768x768xf32>
      ml_program.global_store @WV = %WV_next : tensor<768x768xf32>

      %WO_next = nova.sub %WO_cur, %grad_scaled : tensor<768x768xf32>, tensor<768x768xf32>
      ml_program.global_store @WO = %WO_next : tensor<768x768xf32>

      %W1_next = nova.sub %W1_cur, %grad_scaled : tensor<768x768xf32>, tensor<768x768xf32>
      ml_program.global_store @W1 = %W1_next : tensor<768x768xf32>

      %W2_next = nova.sub %W2_cur, %grad_scaled : tensor<768x768xf32>, tensor<768x768xf32>
      ml_program.global_store @W2 = %W2_next : tensor<768x768xf32>
      
      %bQ_next = nova.sub %bQ_cur, %db_scaled : tensor<768xf32>, tensor<768xf32>
      ml_program.global_store @bQ = %bQ_next : tensor<768xf32>

      %bK_next = nova.sub %bK_cur, %db_scaled : tensor<768xf32>, tensor<768xf32>
      ml_program.global_store @bK = %bK_next : tensor<768xf32>

      %bV_next = nova.sub %bV_cur, %db_scaled : tensor<768xf32>, tensor<768xf32>
      ml_program.global_store @bV = %bV_next : tensor<768xf32>

      %bO_next = nova.sub %bO_cur, %db_scaled : tensor<768xf32>, tensor<768xf32>
      ml_program.global_store @bO = %bO_next : tensor<768xf32>

      %b1_next = nova.sub %b1_cur, %db_scaled : tensor<768xf32>, tensor<768xf32>
      ml_program.global_store @b1 = %b1_next : tensor<768xf32>

      %b2_next = nova.sub %b2_cur, %db_scaled : tensor<768xf32>, tensor<768xf32>
      ml_program.global_store @b2 = %b2_next : tensor<768xf32>
    }
    
    %final_WQ = ml_program.global_load @WQ : tensor<768x768xf32>
    return %final_WQ : tensor<768x768xf32>
  }
}
