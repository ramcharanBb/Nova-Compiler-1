// convert to llvm includes
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// other conversion includes from mlir
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
// utils
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
// buffer includes
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"

// nova dialect includes
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
// optimization passes includes
#include "Compiler/Transforms/CleanupPass.h"
#include "Compiler/Transforms/Affine/AffineFullUnroll.h"
#include "Compiler/Transforms/FastmathFlag.h"
#include "Compiler/Transforms/ParallelizeOuterLoops.h"
#include "Compiler/Transforms/FuseMatmulBias.h"

// gpu
#include "mlir/Dialect/GPU/IR/GPUDialect.h"     // For GPUModuleOp
#include "mlir/Dialect/GPU/Transforms/Passes.h" // For createGpuKernelOutliningPass
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

// lowering passes
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

// header of this file
#include "Compiler/Pipeline/Gpupipeline.h"
#include "Compiler/Transforms/Affine/DependencyAnalysisTestPass.h"

using namespace mlir;

namespace mlir
{
    namespace nova
    {
#define GEN_PASS_REGISTRATION
#include "Compiler/Transforms/Passes.h.inc"
    }
} // namespace mlir::nova

void mlir::nova::registerNovaGPUPipelines()
{
    PassPipelineRegistration<>("nova-gpu-pipeline",
                               "Nova gpu Pipeline",
                               createNovaGPUPipelines);
}

void mlir::nova::createNovaGPUPipelines(OpPassManager &pm)
{
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::nova::createNovaToTosaLoweringPass());
    pm.addNestedPass<func::FuncOp>(createNovaToLinalgLoweringPass());

    // tosa to linalg named ops
    pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    // tosa to linalg
    pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalg());

    // tosa to arith
    pm.addPass(mlir::createTosaToArithPass());
    // tosa to tensor
    pm.addPass(mlir::createTosaToTensorPass());
    // tosa to scf
    pm.addPass(mlir::createTosaToSCFPass());
    pm.addNestedPass<func::FuncOp>(mlir::nova::createFuseMatmulBiasPass());
    pm.addPass(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());

    bufferization::OneShotBufferizePassOptions bufferizeOptions;
    bufferizeOptions.bufferizeFunctionBoundaries = true;
    bufferizeOptions.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;

    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));

    bufferization::BufferDeallocationPipelineOptions deallocationOptions;
    bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

    // pm.addNestedPass<func::FuncOp>(mlir::linalg::createLinalgPromotionPass(0,3));
    //   Lower Linalg to Affine loops
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass());
    // Clean up the scalar/temporary loads created by fusion
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());

    // 3. TILE the single fused loop nest
    if (failed(parsePassPipeline("func.func(affine-loop-tile{tile-sizes=32,32,32})", pm)))
    {
        llvm::errs() << "Failed to parse tiling\n";
    }

    // 4. PARALLELIZE
    parsePassPipeline("func.func(affine-parallelize)", pm);

    // 5. DATA COPY & UNROLL
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineDataCopyGenerationPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopUnrollPass(4));

    //  Simplify affine maps and sets
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());

    // Perform affine vectorization (formerly super-vectorize)
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineVectorize());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::createLowerAffinePass());
    // 6. LOWER TO GPU
    pm.addPass(mlir::createGpuMapParallelLoopsPass());
    pm.addPass(mlir::createConvertParallelLoopToGpuPass());

    // 7. OUTLINE (This will now only see ONE launch region -> ONE module)
    pm.addPass(mlir::createGpuKernelOutliningPass());
    auto &gpuPm = pm.nest<gpu::GPUModuleOp>();

    //convert to nvvm
    gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps());
    gpuPm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    
    //convert to llvm
    pm.addPass(mlir::createGpuToLLVMConversionPass());
     pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    
    pm.addPass(mlir::createArithToLLVMConversionPass());
    // pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
     pm.addPass(createCanonicalizerPass());

}
