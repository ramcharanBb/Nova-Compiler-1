#include "Compiler/Transforms/FuseMatmulBias.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

// Pattern to fuse: matmul + add (bias)
struct FuseMatmulBiasPattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  // Helper to skip through reshaping ops (ExpandShapeOp, CollapseShapeOp)
  Operation* skipReshapes(Value startValue, SmallVectorImpl<Operation*> &reshapes) const {
    Value current = startValue;
    while (true) {
      Operation *defOp = current.getDefiningOp();
      if (!defOp) return nullptr;

      if (isa<tensor::ExpandShapeOp>(defOp) || isa<tensor::CollapseShapeOp>(defOp) || isa<tensor::CastOp>(defOp)) {
        reshapes.push_back(defOp);
        current = defOp->getOperand(0);
      } else {
        return defOp;
      }
    }
  }


  // Construct the inverse reshape op. 
  // If original was Expand [A]->[B], new op is Create Collapse [B]->[A]
  // If original was Collapse [A]->[B], new op is Create Expand [B]->[A]
  Value createInverseReshape(OpBuilder &builder, Location loc, Operation *reshapeOp, Value input) const {
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(reshapeOp)) {
      // Inverse of Expand is Collapse
      // We need to collapse 'input' using the same reassociation indices
      return builder.create<tensor::CollapseShapeOp>(
          loc, expandOp.getSrc().getType(), input, expandOp.getReassociationIndices());
    } else if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(reshapeOp)) {
        // Inverse of Collapse is Expand
        return builder.create<tensor::ExpandShapeOp>(
            loc, collapseOp.getSrc().getType(), input, collapseOp.getReassociationIndices());
    } else if (auto castOp = dyn_cast<tensor::CastOp>(reshapeOp)) {
        // Inverse of Cast A->B is Cast B->A
        // castOp casts Source->Dest. We want to cast input (type Dest) -> Source.
        return builder.create<tensor::CastOp>(loc, castOp.getSource().getType(), input);
    }
    return nullptr;
  }

  LogicalResult matchAndRewrite(GenericOp addOp,
                                  PatternRewriter &rewriter) const override {
    // Check if this is an elementwise add
    if (!isElementwise(addOp))
      return failure();

    // Get the input to the add
    Value addInput1 = addOp.getDpsInputOperand(0)->get();
    Value addInput2 = addOp.getDpsInputOperand(1)->get();

    SmallVector<Operation*> reshapes1, reshapes2;
    Operation* prodOp1 = skipReshapes(addInput1, reshapes1);
    Operation* prodOp2 = skipReshapes(addInput2, reshapes2);

    GenericOp matmulOp;
    Value bias;
    SmallVector<Operation*> activeReshapes;

    // Check if input comes from a matmul
    auto isMatmulOp = [&](Operation* op) {
        return op && isa<GenericOp>(op) && isMatmul(cast<GenericOp>(op));
    };

    // Case 1: prodOp1 is matmul
    if (isMatmulOp(prodOp1) && !isMatmulOp(prodOp2)) {
      matmulOp = cast<GenericOp>(prodOp1);
      bias = addInput2;
      activeReshapes = reshapes1;
    }
    // Case 2: prodOp2 is matmul
    else if (isMatmulOp(prodOp2) && !isMatmulOp(prodOp1)) {
        matmulOp = cast<GenericOp>(prodOp2);
        bias = addInput1;
        activeReshapes = reshapes2;
    } else {
        return failure();
    }
    
    // Check that matmul has only one use (the chain leading to this add)
    // If there are reshapes, the matmul output effectively flows into them.
    if (!matmulOp->hasOneUse()) 
      return failure();
    
    // If there are reshapes, check intermediate uses too?
    // For simplicity, assume the chain is single-use for fusion safety.
    Value currentVal = matmulOp.getResult(0);
    for (int i = activeReshapes.size() - 1; i >= 0; --i) {
        if (!currentVal.hasOneUse()) return failure();
        currentVal = activeReshapes[i]->getResult(0); 
    }

    llvm::errs() << "Found fusible matmul + add pattern (with " << activeReshapes.size() << " reshapes)!\n";
      
    // Get matmul inputs
    Value A = matmulOp.getDpsInputOperand(0)->get();
    Value B = matmulOp.getDpsInputOperand(1)->get();
    Value adda=addOp.getDpsInputOperand(0)->get();
    Value addb=addOp.getDpsInputOperand(1)->get();
    // The bias currently matches the 'add' output shape (which is reshaped from matmul)
    // We need to apply the INVERSE reshapes to the bias to make it match the matmul output.
    Value transformedBias = bias;
    Location loc = addOp.getLoc();

    // Iterate reshapes in reverse (from Matmul out -> Add in).
    for (auto *reshapeOp : activeReshapes) {
        transformedBias = createInverseReshape(rewriter, loc, reshapeOp, transformedBias);
        if(!transformedBias) return failure();
    }

    // The 'transformedBias' tensor becomes the initial output (accumulator).
    // Validate types: The Bias must match the Matmul accumulator type (e.g. f32).
    // If we have f16 bias and f32 matmul result, we must cast the bias to f32.
    auto matmulType = cast<RankedTensorType>(matmulOp.getResult(0).getType());
    auto biasType = cast<RankedTensorType>(transformedBias.getType());

    Value output = transformedBias;

    if (matmulType.getElementType() != biasType.getElementType()) {
        // We need to cast the bias to the matmul type.
        // We use a linalg.generic op to perform the cast because arith.extf on tensors
        // is not supported by one-shot-bufferize.
        
        auto newType = RankedTensorType::get(biasType.getShape(), matmulType.getElementType());
        Value initTensor = rewriter.create<tensor::EmptyOp>(loc, newType.getShape(), newType.getElementType());
        
        SmallVector<AffineMap> maps = {
            rewriter.getMultiDimIdentityMap(biasType.getRank()), // Input
            rewriter.getMultiDimIdentityMap(biasType.getRank()),//input2
         //   rewriter.getMultiDimIdentityMap(biasType.getRank())  // Output
        };
        
        SmallVector<utils::IteratorType> iterators(biasType.getRank(), utils::IteratorType::parallel);
        
        output = rewriter.create<GenericOp>(
            loc,
            newType,
            ValueRange{addInput2},
            ValueRange{initTensor},
            maps,
            iterators,
            [&](OpBuilder &b, Location loc, ValueRange args) {
               //get the body of generic without add op
          Block *addBody = addOp.getBody();
          
           IRMapping mapping;
          mapping.map(addBody->getArgument(0), args[0]);
          mapping.map(addBody->getArgument(1), args[1]);
           //mapping.map(addBody->getArgument(2), args[2]);
          Value castedValue = args[0];
          for (Operation &op : addBody->getOperations()) {
            //if it is add op skip
            if(isa<arith::AddFOp>(op))
              continue;
            if(isa<arith::AddIOp>(op))
              continue;
            if (isa<linalg::YieldOp>(op))
              continue;
            llvm::errs() << "Cloning op in cast generic: " << op << "\n";
            Operation *clonedOp=b.clone(op, mapping);
             if (clonedOp->getNumResults() > 0) {
                castedValue = clonedOp->getResult(0);
            }
         }
  // Value addValue =
  //     mapping.lookup(addBody->getTerminator()->getOperand(0));
  b.create<linalg::YieldOp>(loc, castedValue);
            }
        ).getResult(0);
    }

    // Broadcast the bias if shapes don't match (e.g. 1D bias + 2D matmul)
    // The fused GenericOp requires 'outs' to match the result shape.
    auto outputTensorType = cast<RankedTensorType>(output.getType());
    if (outputTensorType.getShape() != matmulType.getShape()) {
       if (activeReshapes.empty()) {
           // If no reshapes, we can reuse the AddOp's maps to broadcast the bias.
           Value initTensor = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(), matmulType.getElementType());
           
           // Determine which input of AddOp was the bias
           int biasIdx = (bias == addInput2) ? 1 : 0;
           
           AffineMap biasMap = addOp.getIndexingMapsArray()[biasIdx];
           AffineMap outMap = addOp.getIndexingMapsArray()[2];
           
           SmallVector<AffineMap> maps = {biasMap, outMap};
           SmallVector<utils::IteratorType> iterators = 
                llvm::to_vector<4>(addOp.getIteratorTypesArray());
           
           auto broadcastOp = rewriter.create<GenericOp>(
               loc, 
               matmulType, // Result type
               ValueRange{output}, // Input (bias)
               ValueRange{initTensor}, // Output (init)
               maps, 
               iterators,
               [&](OpBuilder &b, Location loc, ValueRange args) {
                   b.create<linalg::YieldOp>(loc, args[0]);
               }
           );
           output = broadcastOp.getResult(0);
       }
       // If reshapes exist, broadcasting is complex. Fall through (might fail verification if shapes mismatch).
    }
    
    // The fused op will produce the shape of the original matmul.
    SmallVector<AffineMap> matmulMaps = 
        llvm::to_vector<4>(matmulOp.getIndexingMapsArray());
    SmallVector<utils::IteratorType> iteratorTypes = 
        llvm::to_vector<4>(matmulOp.getIteratorTypesArray());


    SmallVector<AffineMap> indexingMaps;

    if (matmulMaps.size() != 3) {
      // Basic validation check in case the matmul definition is non-standard
      return failure(); 
    }
    
    indexingMaps.push_back(matmulMaps[0]);
    indexingMaps.push_back(matmulMaps[1]); 
    indexingMaps.push_back(matmulMaps[2]);

    // Create the fused generic op
    auto fusedOp = rewriter.create<GenericOp>(
        loc,
        matmulOp.getResult(0).getType(), 
        ValueRange{A, B},
        ValueRange{output}, 
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = A, args[1] = B, args[2] = Bias/Accum
          
          Block *matmulBody = matmulOp.getBody();
          
          IRMapping mapping;
          // Map matmul body arguments to the new block arguments
          // Matmul body: ^bb0(%a, %b, %c)
          mapping.map(matmulBody->getArgument(0), args[0]);
          mapping.map(matmulBody->getArgument(1), args[1]);
          mapping.map(matmulBody->getArgument(2), args[2]);
          
  // ---------------------------------------------------------
  // 1. Clone MATMUL BODY
  // ---------------------------------------------------------
  for (Operation &op : matmulBody->getOperations()) {
    if (isa<linalg::YieldOp>(op))
      continue;
    b.clone(op, mapping);
  }

  // The matmul reduction result:
  Value matmulValue =
      mapping.lookup(matmulBody->getTerminator()->getOperand(0));

  // The matmulValue (result of prod + args[2]) is the fused result.
  // We do NOT inline the AddOp body again because we already incorporated the bias 
  // via the accumulator initialization.
  b.create<linalg::YieldOp>(loc, matmulValue);
  }
    );

    // we have the fused result in the Matmul shape.
    // We need to reshape it BACK to the Add shape to replace the Add op.
    Value fusedResult = fusedOp.getResult(0);
    
    for (int i = activeReshapes.size() - 1; i >= 0; --i) {
        Operation* op = activeReshapes[i];
        if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
            // Apply Expand
             fusedResult = rewriter.create<tensor::ExpandShapeOp>(
                loc, expand.getResult().getType(), fusedResult, expand.getReassociationIndices());
        } else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
             fusedResult = rewriter.create<tensor::CollapseShapeOp>(
                loc, collapse.getResult().getType(), fusedResult, collapse.getReassociationIndices());
        } else if (auto castOp = dyn_cast<tensor::CastOp>(op)) {
             fusedResult = rewriter.create<tensor::CastOp>(
                loc, castOp.getResult().getType(), fusedResult);
        }
    }

    llvm::errs() << "Successfully fused matmul + bias into single operation (thru reshapes)!\n";

    // Replace the original add operation with the new fused (and reshaped) operation
    rewriter.replaceOp(addOp, fusedResult);
    // The matmul op and reshapes will be automatically removed if dead
    return success();
  }

private:
  bool isElementwise(GenericOp op) const {
    //check if the operatiopn has two inputs
    if (op.getNumDpsInputs() != 2)
      return false;
    //return true for only add operation
    if(auto arithOp = dyn_cast<arith::AddFOp>(op.getBody()->getTerminator()->getOperand(0).getDefiningOp())) {
        return true; 
    }
    return false;
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