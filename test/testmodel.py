import torch
import torch.nn.functional as F

# Configuration exactly matching testmodel.mlir
hidden_dim = 768
seq_len = 128
num_iterations = 5
lr = 0.01
scale = 0.036  # ≈ 1/sqrt(hidden_dim)

def train_transformer():
    # 1. Initial Weights Setup
    # MLIR uses nova.rndm2d with seed 42 and range [-0.1, 0.1]
    torch.manual_seed(42)
    
    # Weights (hidden_dim x hidden_dim)
    WQ = (torch.rand(hidden_dim, hidden_dim) * 0.2 - 0.1).detach().requires_grad_(True)
    WK = (torch.rand(hidden_dim, hidden_dim) * 0.2 - 0.1).detach().requires_grad_(True)
    WV = (torch.rand(hidden_dim, hidden_dim) * 0.2 - 0.1).detach().requires_grad_(True)
    WO = (torch.rand(hidden_dim, hidden_dim) * 0.2 - 0.1).detach().requires_grad_(True)
    W1 = (torch.rand(hidden_dim, hidden_dim) * 0.2 - 0.1).detach().requires_grad_(True)
    W2 = (torch.rand(hidden_dim, hidden_dim) * 0.2 - 0.1).detach().requires_grad_(True)

    # Biases (initialized to 0.0 in MLIR)
    bQ = torch.zeros(hidden_dim, requires_grad=True)
    bK = torch.zeros(hidden_dim, requires_grad=True)
    bV = torch.zeros(hidden_dim, requires_grad=True)
    bO = torch.zeros(hidden_dim, requires_grad=True)
    b1 = torch.zeros(hidden_dim, requires_grad=True)
    b2 = torch.zeros(hidden_dim, requires_grad=True)

    params = [WQ, WK, WV, WO, W1, W2, bQ, bK, bV, bO, b1, b2]

    print(f"Starting Training ({num_iterations} iterations)...")

    for i in range(num_iterations):
        # 2.1. Batch Preparation
        # Mocking the nova.rndm2d behavior per iteration
        X = (torch.rand(seq_len, hidden_dim) * 0.2 - 0.1)
        labels = (torch.rand(seq_len, hidden_dim) * 0.2 - 0.1)

        # 2.2. Forward Pass
        # Self-Attention
        Q = torch.matmul(X, WQ) + bQ
        K = torch.matmul(X, WK) + bK
        V = torch.matmul(X, WV) + bV

        scores = torch.matmul(Q, K.transpose(0, 1)) * scale
        probs = F.softmax(scores, dim=1)
        context = torch.matmul(probs, V)
        
        attn_out = torch.matmul(context, WO) + bO
        x_attn = X + attn_out

        # Feed-Forward
        ff1 = torch.matmul(x_attn, W1) + b1
        ff1_act = F.relu(ff1)
        preds = torch.matmul(ff1_act, W2) + b2

        # 2.3. Backward Pass (Simplified Manual Gradients as per MLIR logic)
        # Note: MLIR uses a simplified gradient calculation: 
        # dW = X^T * (preds - labels)
        # db = sum(preds - labels, dim=0)
        
        diff = preds - labels
        XT = X.transpose(0, 1)
        
        # In the MLIR, all parameters use the same simplified dW_common and db_common
        dW_common = torch.matmul(XT, diff)
        db_common = torch.sum(diff, dim=0)

        # 2.4. Batch Update (SGD)
        with torch.no_grad():
            for p in params:
                if p.dim() == 2: # Weight
                    p -= lr * dW_common
                else: # Bias
                    p -= lr * db_common

        print(f"Iteration {i+1} complete.")

    print("\nFinal WQ (Sample 5x5):")
    print(WQ[:5, :5])
    return WQ

if __name__ == "__main__":
    train_transformer()
