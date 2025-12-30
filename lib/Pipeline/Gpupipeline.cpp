// convert to llvm includes
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
// other conversion includes from mlir
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
// utils
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
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
#include "Compiler/Transforms/FuseMatmulBias.h"

// gpu
#include "mlir/Dialect/GPU/IR/GPUDialect.h"     // For GPUModuleOp
#include "mlir/Dialect/GPU/Transforms/Passes.h" // For createGpuKernelOutliningPass
#include "mlir/Dialect/GPU/Pipelines/Passes.h"  // For GpuNVVMAttachTarget
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

// lowering passes
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

// header of this file
#include "Compiler/Pipeline/Gpupipeline.h"

using namespace mlir;

namespace mlir {
namespace nova {
    void createNovaGPUPipelines(mlir::OpPassManager &pm) {
    // 1. NOVA TO TOSA/LINALG
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::nova::createNovaToTosaLoweringPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createNovaToLinalgLoweringPass());

    // 2. TOSA TO LINALG (Named and regular)
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

    // 3. TOSA TO ARITH/TENSOR/SCF
    pm.addPass(mlir::createTosaToArithPass());
    pm.addPass(mlir::createTosaToTensorPass());
    pm.addPass(mlir::createTosaToSCFPass());

    // 4. NOVA TRANSFORMS & LINALG OPT
    pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createFuseMatmulBiasPass());
    pm.addPass(createCanonicalizerPass());
    // Tiling is handled in Section 6 after parallel loop conversion
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgGeneralizeNamedOpsPass());

    // 5. BUFFERIZATION & DEALLOCATION
    bufferization::OneShotBufferizePassOptions bufferizeOptions;
    bufferizeOptions.bufferizeFunctionBoundaries = true;
    bufferizeOptions.functionBoundaryTypeConversion = bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));
    bufferization::BufferDeallocationPipelineOptions deallocationOptions;
    bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

    // 6. LINALG TO PARALLEL LOOPS
    pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopFusionPass());
    pm.addPass(mlir::createParallelLoopTilingPass({32, 32, 32}));
    pm.addPass(createCanonicalizerPass());

    // 7. (SKIPPED) AFFINE OPTIMIZATIONS
    // pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass());
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());

    // // Tiling
    // if (failed(parsePassPipeline("func.func(affine-loop-tile{tile-sizes=32,32,32})", pm))) {
    //     llvm::errs() << "Failed to parse tiling\n";
    // }

    // // Parallelize
    // if (failed(parsePassPipeline("func.func(affine-parallelize)", pm))) {
    //     llvm::errs() << "Failed to parse parallelize\n";
    // }

    // // Data Copy & Unroll
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineDataCopyGenerationPass());
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopUnrollPass(4));

    // // Simplify & Vectorize
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineVectorize());
    // pm.addPass(createCanonicalizerPass());
    // pm.addPass(mlir::createLowerAffinePass());

    // 8. LOWER TO GPU
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createGpuMapParallelLoopsPass());
    pm.addPass(mlir::createConvertParallelLoopToGpuPass());

    // 9. GPU LOWERING CONTEXT & BINARY GENERATION (Stage 2)
    pm.addPass(mlir::createGpuDecomposeMemrefsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());

    mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
    nvvmTargetOptions.chip = "sm_70";
    nvvmTargetOptions.features = "+ptx70";
    nvvmTargetOptions.linkLibs.push_back("/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc");
    pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

    auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
    gpuPm.addPass(mlir::createLowerAffinePass());
    gpuPm.addPass(mlir::createSCFToControlFlowPass());
    gpuPm.addPass(mlir::createCanonicalizerPass());
    gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps());
    gpuPm.addPass(mlir::createArithToLLVMConversionPass());
    gpuPm.addPass(mlir::createConvertMathToLLVMPass());
    gpuPm.addPass(mlir::createConvertIndexToLLVMPass());
    gpuPm.addPass(mlir::createConvertControlFlowToLLVMPass());
    gpuPm.addPass(mlir::createCanonicalizerPass());
    gpuPm.addPass(mlir::createCSEPass());
    gpuPm.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    pm.addPass(mlir::createGpuToLLVMConversionPass());
    
    mlir::GpuModuleToBinaryPassOptions binaryOptions;
    binaryOptions.toolkitPath = "/usr/local/cuda-12.1";
    pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));
    pm.addPass(createCanonicalizerPass());
    
    // 10. FINAL LOWERING TO LLVM
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void registerNovaGPUPipelines() {
  PassPipelineRegistration<>("nova-gpu-pipeline",
                            "Nova GPU Pipeline",
                            createNovaGPUPipelines);
}

} // namespace nova
} // namespace mlir
