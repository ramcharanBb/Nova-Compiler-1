#include "Compiler/Transforms/FuseMatmulBias.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

// Pattern to fuse: matmul + add (bias)
struct FuseMatmulBiasPattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp addOp,
                                  PatternRewriter &rewriter) const override {
    // Check if this is an elementwise add
    if (!isElementwise(addOp))
      return failure();

    // Get the input to the add
    Value addInput1 = addOp.getDpsInputOperand(0)->get();
    Value addInput2 = addOp.getDpsInputOperand(1)->get();

    // Check if input comes from a matmul
    GenericOp matmulOp;
    auto matmulOp1 = addInput1.getDefiningOp<GenericOp>();
    auto matmulOp2 = addInput2.getDefiningOp<GenericOp>();
    Value bias;
// Case 1: only input1 is matmul
if (matmulOp1 && isMatmul(matmulOp1) &&
    !(matmulOp2 && isMatmul(matmulOp2))) {
  matmulOp = matmulOp1;
  bias = addInput2;
}
// Case 2: only input2 is matmul
else if (matmulOp2 && isMatmul(matmulOp2) &&
         !(matmulOp1 && isMatmul(matmulOp1))) {
  matmulOp = matmulOp2;
  bias = addInput1;
} else {
  return failure();
}
    // Check that matmul has only one use (this add)
    if (!matmulOp || !matmulOp->hasOneUse()) 
      return failure();
    llvm::errs() << "Found fusible matmul + add pattern!\n";
      
    // Get matmul inputs
    Value A = matmulOp.getDpsInputOperand(0)->get();
    Value B = matmulOp.getDpsInputOperand(1)->get();

    // The 'bias' tensor becomes the initial output (accumulator).
    Value output = bias;

    // Get location for the new operation
    Location loc = addOp.getLoc();

    // The result type is the type of the Add operation
    auto resultType = cast<RankedTensorType>(addOp.getResult(0).getType()); 
    SmallVector<AffineMap> matmulMaps = 
        llvm::to_vector<4>(matmulOp.getIndexingMapsArray());
    SmallVector<utils::IteratorType> iteratorTypes = 
        llvm::to_vector<4>(matmulOp.getIteratorTypesArray());


    SmallVector<AffineMap> indexingMaps;

    // We only need the first two maps (A, B access) and the third map (Output C access)
    // from the original matmul operation.
    if (matmulMaps.size() != 3) {
      // Basic validation check in case the matmul definition is non-standard
      return failure(); 
    }
    
    indexingMaps.push_back(matmulMaps[0]); // A Access Map
    indexingMaps.push_back(matmulMaps[1]); // B Access Map
    indexingMaps.push_back(matmulMaps[2]);

    // Create the fused generic op
    auto fusedOp = rewriter.create<GenericOp>(
        loc,
        resultType,
        ValueRange{A, B},     // Inputs are only A and B
        ValueRange{output},   // Output is the bias tensor
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = A[i,k], args[1] = B[k,j], args[2] = C[i,j] (initial value is bias)
          
          // Loop body is ONLY matmul accumulation (C = C + A * B)
          Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
          
          //if the input one is matmul output
          if(matmulOp)
            mul = b.create<arith::AddFOp>(loc, args[2], mul);
          else
            //if the input two is matmul output
            mul= b.create<arith::AddFOp>(loc, args[1], mul);
          b.create<linalg::YieldOp>(loc, mul);
        }
    );

    llvm::errs() << "Successfully fused matmul + bias into single operation!\n";

    // Replace the original add operation with the new fused operation
    rewriter.replaceOp(addOp, fusedOp.getResult(0));

    // The matmul op will be automatically removed since it has no more uses

    return success();
  }

private:
  bool isElementwise(GenericOp op) const {
    //check if the operatiopn has two inputs
    if (op.getNumDpsInputs() != 2)
      return false;
    // Check if all loops are parallel
    return llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType it) {
      return it == utils::IteratorType::parallel;
    });
  }

  bool isMatmul(GenericOp op) const {
    auto iterators = op.getIteratorTypesArray();
    // Matmul has 2 parallel + 1 reduction dimension
    if (iterators.size() < 3)
      return false;

    // We need exactly one reduction loop
    int reductionCount = 0;
    for (auto it : iterators) {
      if (it == utils::IteratorType::reduction) {
        reductionCount++;
      }
    }
    if (reductionCount != 1) return false;

    // Ensure all other loops are parallel
    return llvm::all_of(iterators, [](utils::IteratorType it) {
      return it == utils::IteratorType::parallel || it == utils::IteratorType::reduction;
    });
  }
};

// The pass that applies the fusion pattern
struct FuseMatmulBiasPass
    : public PassWrapper<FuseMatmulBiasPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatmulBiasPass)

  void runOnOperation() override {
    auto func = getOperation();

    llvm::errs() << "=== Running FuseMatmulBias Pass ===\n";

    // Apply the pattern
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulBiasPattern>(&getContext());

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }

    llvm::errs() << "=== FuseMatmulBias Pass Complete ===\n";
  }

  StringRef getArgument() const final { return "fuse-matmul-bias"; }

  StringRef getDescription() const final {
    return "Fuse matmul + bias add into a single operation";
  }
};

} // namespace

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createFuseMatmulBiasPass() {
  return std::make_unique<FuseMatmulBiasPass>();
}

} // namespace nova
} // namespace mlir