#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "Compiler/Dialect/nova/NovaDialect.h"

using namespace mlir;

namespace mlir {
namespace nova {

static bool isMemorySpaceOne(Attribute memorySpace) {
  if (!memorySpace) return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
    return intAttr.getInt() == 1;
  }
  return false;
}

class ConvertAllocOp : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op, PatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (!isMemorySpaceOne(type.getMemorySpace())) return failure();

    rewriter.replaceOpWithNewOp<gpu::AllocOp>(
        op, type, /*asyncToken=*/Type(), /*asyncDependencies=*/ValueRange{},
        op.getDynamicSizes(), op.getSymbolOperands(), /*hostShared=*/false);
    return success();
  }
};

class ConvertDeallocOp : public OpRewritePattern<memref::DeallocOp> {
public:
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op, PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();
    MemRefType type = llvm::dyn_cast<MemRefType>(memref.getType());
    if (!type || !isMemorySpaceOne(type.getMemorySpace())) return failure();

    rewriter.replaceOpWithNewOp<gpu::DeallocOp>(op, TypeRange{}, ValueRange{}, memref);
    return success();
  }
};

struct ConvertMemRefToGpuPass
    : public PassWrapper<ConvertMemRefToGpuPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMemRefToGpuPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Replace #nova.device<"1"> with 1 : i32 everywhere
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](nova::NovaDeviceAttr attr) -> std::optional<Attribute> {
      if (attr.getValue().getValue() == "1")
        return IntegerAttr::get(IntegerType::get(ctx, 32), 1);
      return std::nullopt;
    });
    
    replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    // 2. Convert memref.alloc/dealloc with memory space 1 to gpu.alloc/dealloc
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAllocOp, ConvertDeallocOp>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "convert-memref-to-gpu"; }
  StringRef getDescription() const final {
    return "Convert memref.alloc/dealloc with memory space 1 to gpu.alloc/dealloc and legalize types";
  }
};

std::unique_ptr<Pass> createConvertMemRefToGpuPass() {
  return std::make_unique<ConvertMemRefToGpuPass>();
}

} // namespace nova
} // namespace mlir
