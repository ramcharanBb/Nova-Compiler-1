#include "Compiler/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace mlir {
namespace nova {

// Helper to declare runtime functions
static LLVM::LLVMFuncOp getOrDeclareFunc(ModuleOp module, OpBuilder &rewriter, StringRef name, Type resultTy, ArrayRef<Type> argTypes) {
    if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return func;
    
    // Check for collision
    if (auto existing = SymbolTable::lookupSymbolIn(module, name)) {
        if (auto llvmFunc = llvm::dyn_cast<LLVM::LLVMFuncOp>(existing))
            return llvmFunc;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcType = LLVM::LLVMFunctionType::get(resultTy, argTypes);
    return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, funcType);
}

class ConvertGpuAllocToCall : public OpRewritePattern<gpu::AllocOp> {
public:
  using OpRewritePattern<gpu::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::AllocOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    
    auto voidPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMalloc = getOrDeclareFunc(module, rewriter, "cudaMalloc", int32Ty, {voidPtrTy, int64Ty});

    MemRefType memRefType = op.getType();
    int64_t elementSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elementSize == 0) elementSize = 1;

    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));
    
    unsigned dynamicIdx = 0;
    auto dynamicSizes = op.getDynamicSizes();
    
    for (int i = 0; i < memRefType.getRank(); ++i) {
        Value dimSize;
        if (memRefType.isDynamicDim(i)) {
             Value dynSize = dynamicSizes[dynamicIdx++];
             dimSize = rewriter.create<UnrealizedConversionCastOp>(loc, int64Ty, dynSize).getResult(0);
        } else {
             dimSize = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
        }
        sizeBytes = rewriter.create<LLVM::MulOp>(loc, sizeBytes, dimSize);
    }

    Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, rewriter.getI32IntegerAttr(1));
    Value ptrVar = rewriter.create<LLVM::AllocaOp>(loc, voidPtrTy, voidPtrTy, one, 8);
    
    rewriter.create<LLVM::CallOp>(loc, cudaMalloc, ValueRange{ptrVar, sizeBytes});
    Value devicePtr = rewriter.create<LLVM::LoadOp>(loc, voidPtrTy, ptrVar);

    SmallVector<Type> elemTypes;
    elemTypes.push_back(voidPtrTy); 
    elemTypes.push_back(voidPtrTy);
    elemTypes.push_back(int64Ty);
    if (memRefType.getRank() > 0) {
        elemTypes.push_back(LLVM::LLVMArrayType::get(int64Ty, memRefType.getRank()));
        elemTypes.push_back(LLVM::LLVMArrayType::get(int64Ty, memRefType.getRank()));
    }
    Type descType = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);

    Value desc = rewriter.create<LLVM::UndefOp>(loc, descType);
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, devicePtr, ArrayRef<int64_t>{0});
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, devicePtr, ArrayRef<int64_t>{1});
    Value zero = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, rewriter.getI64IntegerAttr(0));
    desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, zero, ArrayRef<int64_t>{2});

    if (memRefType.getRank() > 0) {
        Value sizesArray = rewriter.create<LLVM::UndefOp>(loc, elemTypes[3]);
        dynamicIdx = 0;
        for (int i=0; i<memRefType.getRank(); ++i) {
             Value dimSize;
             if (memRefType.isDynamicDim(i)) {
                 Value dynSize = dynamicSizes[dynamicIdx++];
                 dimSize = rewriter.create<UnrealizedConversionCastOp>(loc, int64Ty, dynSize).getResult(0);
             } else {
                 dimSize = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
             }
             sizesArray = rewriter.create<LLVM::InsertValueOp>(loc, sizesArray, dimSize, ArrayRef<int64_t>{i});
        }
        desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, sizesArray, ArrayRef<int64_t>{3});
        Value stridesArray = rewriter.create<LLVM::UndefOp>(loc, elemTypes[4]);
        desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, stridesArray, ArrayRef<int64_t>{4});
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, memRefType, desc);
    return success();
  }
};

class ConvertGpuMemcpyToCall : public OpRewritePattern<gpu::MemcpyOp> {
public:
  using OpRewritePattern<gpu::MemcpyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::MemcpyOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    
    auto voidPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int64Ty = IntegerType::get(ctx, 64);
    auto int32Ty = IntegerType::get(ctx, 32);

    auto cudaMemcpy = getOrDeclareFunc(module, rewriter, "cudaMemcpy", int32Ty, {voidPtrTy, voidPtrTy, int64Ty, int32Ty});

    Value dst = op.getDst();
    Value src = op.getSrc();
    
    auto getPtrFromMemRef = [&](Value val) -> Value {
        auto memRefTy = llvm::cast<MemRefType>(val.getType());
        SmallVector<Type> elemTypes;
        elemTypes.push_back(voidPtrTy); 
        elemTypes.push_back(voidPtrTy);
        elemTypes.push_back(int64Ty);
        if (memRefTy.getRank() > 0) {
             elemTypes.push_back(LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
             elemTypes.push_back(LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
        }
        Type descTy = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);
        Value desc = rewriter.create<UnrealizedConversionCastOp>(loc, descTy, val).getResult(0);
        return rewriter.create<LLVM::ExtractValueOp>(loc, desc, ArrayRef<int64_t>{1}); 
    };

    Value dstPtr = getPtrFromMemRef(dst);
    Value srcPtr = getPtrFromMemRef(src);
    
    MemRefType memRefType = llvm::cast<MemRefType>(dst.getType());
    int64_t elementSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elementSize == 0) elementSize = 1;
    
    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, rewriter.getI64IntegerAttr(elementSize));
    for (int i = 0; i < memRefType.getRank(); ++i) {
         Value dimSize;
         if (memRefType.isDynamicDim(i)) {
             dimSize = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, rewriter.getI64IntegerAttr(1)); 
         } else {
             dimSize = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, rewriter.getI64IntegerAttr(memRefType.getDimSize(i)));
         }
         sizeBytes = rewriter.create<LLVM::MulOp>(loc, sizeBytes, dimSize);
    }
    
    Value kind = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, rewriter.getI32IntegerAttr(4));
    rewriter.create<LLVM::CallOp>(loc, cudaMemcpy, ValueRange{dstPtr, srcPtr, sizeBytes, kind});
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertGpuDeallocToCall : public OpRewritePattern<gpu::DeallocOp> {
public:
  using OpRewritePattern<gpu::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::DeallocOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    auto voidPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto int32Ty = IntegerType::get(ctx, 32);
    auto int64Ty = IntegerType::get(ctx, 64);

    auto cudaFree = getOrDeclareFunc(module, rewriter, "cudaFree", int32Ty, {voidPtrTy});
    auto memRef = op.getMemref();
    
    auto memRefTy = llvm::cast<MemRefType>(memRef.getType());
    SmallVector<Type> elemTypes;
    elemTypes.push_back(voidPtrTy); 
    elemTypes.push_back(voidPtrTy);
    elemTypes.push_back(int64Ty);
    if (memRefTy.getRank() > 0) {
         elemTypes.push_back(LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
         elemTypes.push_back(LLVM::LLVMArrayType::get(int64Ty, memRefTy.getRank()));
    }
    Type descTy = LLVM::LLVMStructType::getLiteral(ctx, elemTypes);
    
    Value desc = rewriter.create<UnrealizedConversionCastOp>(loc, descTy, memRef).getResult(0);
    Value ptr = rewriter.create<LLVM::ExtractValueOp>(loc, desc, ArrayRef<int64_t>{1});
    
    rewriter.create<LLVM::CallOp>(loc, cudaFree, ValueRange{ptr});
    rewriter.eraseOp(op);
    return success();
  }
};

class FixGpuLaunchPass : public PassWrapper<FixGpuLaunchPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixGpuLaunchPass)

    void fixGlobalDtors(ModuleOp module) {
        auto globalDtors = module.lookupSymbol<LLVM::GlobalOp>("llvm.global_dtors");
        if (!globalDtors) {
            return;
        }

        auto initialValue = globalDtors.getValue();
        if (!initialValue) {
            return;
        }

        auto arrayAttr = llvm::dyn_cast<ArrayAttr>(*initialValue);
        if (!arrayAttr) {
            return;
        }

        OpBuilder builder(module.getContext());
        SmallVector<Attribute, 4> newElements;
        bool changed = false;

        for (auto attr : arrayAttr) {
            auto structAttr = llvm::dyn_cast<ArrayAttr>(attr);
            if (structAttr && structAttr.size() >= 2) {
                std::string attrStr;
                {
                    llvm::raw_string_ostream os(attrStr);
                    structAttr[1].print(os);
                }
                
                if (attrStr.find("unload") != std::string::npos) {
                    SmallVector<Attribute, 3> fields;
                    fields.push_back(builder.getI32IntegerAttr(65535));
                    fields.push_back(structAttr[1]);
                    if (structAttr.size() > 2)
                        fields.push_back(structAttr[2]);
                    
                    newElements.push_back(builder.getArrayAttr(fields));
                    changed = true;
                    continue;
                }
            }
            newElements.push_back(attr);
        }

        if (changed) {
            globalDtors.setValueAttr(builder.getArrayAttr(newElements));
        }
    }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    fixGlobalDtors(module);
    
    if (auto existing = SymbolTable::lookupSymbolIn(module, "dealloc_helper")) {
        if (!llvm::isa<LLVM::LLVMFuncOp>(existing)) {
            existing->erase();
        }
    }

    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("dealloc_helper")) {
        OpBuilder builder(module.getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto voidTy = LLVM::LLVMVoidType::get(module.getContext());
        auto ptrTy = LLVM::LLVMPointerType::get(module.getContext());
        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy});
        builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "dealloc_helper", funcTy);
    }

    RewritePatternSet patterns(module.getContext());
    patterns.add<ConvertGpuAllocToCall>(module.getContext());
    patterns.add<ConvertGpuMemcpyToCall>(module.getContext());
    patterns.add<ConvertGpuDeallocToCall>(module.getContext());
    
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createFixGpuLaunchPass() {
  return std::make_unique<FixGpuLaunchPass>();
}

} // namespace nova
} // namespace mlir
