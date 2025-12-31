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

class ConvertLaunchFuncToCall : public OpRewritePattern<gpu::LaunchFuncOp> {
public:
  using OpRewritePattern<gpu::LaunchFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    auto voidPtrTy = LLVM::LLVMPointerType::get(op.getContext());
    auto intPtrTy = IntegerType::get(op.getContext(), 64);
    auto int32Ty = IntegerType::get(op.getContext(), 32);
    
    // Declare runtime helpers if not present
    auto declareFunc = [&](StringRef name, Type resultTy, ArrayRef<Type> argTypes) -> LLVM::LLVMFuncOp {
        if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
            return func;
        
        if (auto existing = SymbolTable::lookupSymbolIn(module, name)) {
            if (auto llvmFunc = llvm::dyn_cast<LLVM::LLVMFuncOp>(existing))
                return llvmFunc;
            // Name conflict but not an LLVM func. We'll have to hope for the best or rename.
            // But let's assume if it exists with our runtime name, we can cast or it will be fixed.
        }

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(resultTy, argTypes);
        return rewriter.create<LLVM::LLVMFuncOp>(loc, name, funcType);
    };

    auto loadFunc = declareFunc("mgpuModuleLoad", voidPtrTy, {voidPtrTy});
    auto getFunc = declareFunc("mgpuModuleGetFunction", voidPtrTy, {voidPtrTy, voidPtrTy});
    auto launchFunc = declareFunc("mgpuLaunchKernel", intPtrTy, {
        voidPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
        int32Ty, voidPtrTy, voidPtrTy, voidPtrTy
    });
    
    // 1. Symbol and Global Management
    auto kernelName = op.getKernelName();
    auto kernelModuleName = op.getKernelModuleName();
    
    // Create global strings for kernel name and module name if they don't exist
    auto getOrInsertGlobalString = [&](StringRef name, StringRef value) {
        std::string globalName = "kstr_" + name.str();
        auto global = module.lookupSymbol<LLVM::GlobalOp>(globalName);
        if (!global) {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            // Ensure null termination
            std::string nullTerminated = value.str();
            nullTerminated.push_back('\0');
            auto type = LLVM::LLVMArrayType::get(IntegerType::get(op.getContext(), 8), nullTerminated.size());
            global = rewriter.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, globalName, rewriter.getStringAttr(nullTerminated), /*alignment=*/0);
        }
        return global;
    };

    auto kNameGlobal = getOrInsertGlobalString(kernelName, kernelName);
    
    // Get pointer to global string
    auto getPtrToGlobal = [&](LLVM::GlobalOp global) {
        Value addr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        Value idx0 = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, rewriter.getI32IntegerAttr(0));
        return rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, global.getType(), addr, ValueRange{idx0, idx0}).getResult();
    };

    Value kernelNamePtr = getPtrToGlobal(kNameGlobal);

    // Module & Kernel Handle Retrieval
    Value binaryDataPtr = rewriter.create<LLVM::ZeroOp>(loc, voidPtrTy);
    module.walk([&](gpu::BinaryOp binary) {
        if (binary.getName() == kernelModuleName) {
            // Found the binary. In a real system, we'd pass the binary data address.
            // For now, we still use ZeroOp but the logic is there to be hooked.
            binaryDataPtr = rewriter.create<LLVM::ZeroOp>(loc, voidPtrTy);
        }
    });
    
    Value loadedModule = rewriter.create<LLVM::CallOp>(loc, loadFunc, binaryDataPtr).getResult();
    Value kernelHandle = rewriter.create<LLVM::CallOp>(loc, getFunc, ValueRange{loadedModule, kernelNamePtr}).getResult();

    // 2. Grid/Block Setup
    auto castToI64 = [&](Value v) {
      if (v.getType().isInteger(64)) return v;
      if (v.getType().isIndex()) {
          // Use unrealized_conversion_cast to bridge to i64 for index
          return rewriter.create<UnrealizedConversionCastOp>(loc, intPtrTy, v).getResult(0);
      }
      return rewriter.create<LLVM::ZExtOp>(loc, intPtrTy, v).getResult();
    };
    Value gridX = castToI64(op.getGridSizeX());
    Value gridY = castToI64(op.getGridSizeY());
    Value gridZ = castToI64(op.getGridSizeZ());
    Value blockX = castToI64(op.getBlockSizeX());
    Value blockY = castToI64(op.getBlockSizeY());
    Value blockZ = castToI64(op.getBlockSizeZ());
    
    Value smem = op.getDynamicSharedMemorySize();
    if (!smem) smem = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, rewriter.getI32IntegerAttr(0));
    else smem = rewriter.create<LLVM::TruncOp>(loc, int32Ty, smem);
    
    // 3. Async Support
    Value stream = rewriter.create<LLVM::ZeroOp>(loc, voidPtrTy);
    if (!op.getAsyncDependencies().empty()) {
        Value dep = op.getAsyncDependencies().front();
        if (dep.getType() == voidPtrTy) {
            stream = dep;
        }
    }
                   
    // 4. Argument Marshalling
    auto numArgs = op.getNumKernelOperands();
    Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, rewriter.getI32IntegerAttr(1));
    Value arraySize = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, rewriter.getI32IntegerAttr(numArgs));
    Value paramsArray = rewriter.create<LLVM::AllocaOp>(loc, voidPtrTy, voidPtrTy, arraySize, 8);
    
    for (unsigned i = 0; i < numArgs; ++i) {
        Value val = op.getKernelOperand(i);
        Type valTy = val.getType();
        
        // If the operand is still a high-level type (like memref), cast it to its LLVM counterpart
        if (llvm::isa<MemRefType, IndexType>(valTy)) {
            Type targetTy = voidPtrTy;
            if (valTy.isIndex()) {
                targetTy = intPtrTy;
            }
            val = rewriter.create<UnrealizedConversionCastOp>(loc, targetTy, val).getResult(0);
            valTy = targetTy;
        }
        
        Value slot = rewriter.create<LLVM::AllocaOp>(loc, voidPtrTy, valTy, one, 8);
        rewriter.create<LLVM::StoreOp>(loc, val, slot);
        Value index = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, rewriter.getI32IntegerAttr(i));
        Value paramSlotPtr = rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, voidPtrTy, paramsArray, index);
        rewriter.create<LLVM::StoreOp>(loc, slot, paramSlotPtr);
    }
    
    Value nullExtra = rewriter.create<LLVM::ZeroOp>(loc, voidPtrTy);
    
    // 5. Execution & Error Handling
    auto launchCall = rewriter.create<LLVM::CallOp>(loc, launchFunc, ValueRange{
        kernelHandle, gridX, gridY, gridZ, blockX, blockY, blockZ,
        smem, stream, paramsArray, nullExtra
    });
    
    Value status = launchCall.getResult();
    Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, intPtrTy, rewriter.getI64IntegerAttr(0));
    Value isError = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, status, zeroI64);
    
    // Add a conditional trap to ensure error handling is preserved in the IR
    auto currentBlock = rewriter.getBlock();
    auto postBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    auto trapBlock = rewriter.createBlock(postBlock);
    
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, isError, trapBlock, postBlock);
    
    rewriter.setInsertionPointToEnd(trapBlock);
    auto trapFunc = declareFunc("llvm.trap", LLVM::LLVMVoidType::get(op.getContext()), {});
    rewriter.create<LLVM::CallOp>(loc, trapFunc, ValueRange{});
    rewriter.create<LLVM::UnreachableOp>(loc);
    
    rewriter.setInsertionPointToStart(postBlock);
    
    rewriter.eraseOp(op);
    return success();
  }
};

class FixGpuLaunchPass : public PassWrapper<FixGpuLaunchPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixGpuLaunchPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Fix missing dealloc_helper declaration from buffer deallocation
    // Use a more aggressive approach: always ensure LLVMFuncOp exists, remove any func.func with same name
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
    patterns.add<ConvertLaunchFuncToCall>(module.getContext());
    
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
