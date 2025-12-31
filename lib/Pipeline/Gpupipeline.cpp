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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

// nova dialect includes
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
// optimization passes includes
#include "Compiler/Transforms/FuseMatmulBias.h"
#include "Compiler/Transforms/FixGpuLaunch.h"

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
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

// header of this file
#include "Compiler/Pipeline/Gpupipeline.h"

using namespace mlir;

namespace mlir
{
    namespace nova
    {
        namespace {
           // Custom passes removed in favor of standard MLIR passes.
        }

        void createNovaGPUPipelines(mlir::OpPassManager &pm)
        {
            pm.addPass(mlir::createCanonicalizerPass());
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
            pm.addPass(mlir::createCanonicalizerPass());
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
            pm.addPass(mlir::createConvertBufferizationToMemRefPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());

            // 6. LINALG TO PARALLEL LOOPS
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgFoldUnitExtentDimsPass());
            pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopFusionPass());
            
            // NEW STRATEGY: Tile First, Then Collapse.
            // 1. Tile: Create Thread Blocks
            pm.addPass(mlir::createParallelLoopTilingPass({8, 8, 8, 1, 1, 1}));
            pm.addPass(mlir::createCanonicalizerPass());

            // 2. Mapping: We map the tiled loops directly to GPU Grid/Block dimensions
            //    using the standard GpuMapParallelLoopsPass.
            //    The 6 loops from tiling map to Grid{X,Y,Z} and Block{X,Y,Z}.


            // // 7. (SKIPPED) AFFINE OPTIMIZATIONS
            // // pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
            // // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass());
            // // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());

            // // // Tiling
            // // if (failed(parsePassPipeline("func.func(affine-loop-tile{tile-sizes=32,32,32})", pm))) {
            // //     llvm::errs() << "Failed to parse tiling\n";
            // // }

            // // // Parallelize
            // // if (failed(parsePassPipeline("func.func(affine-parallelize)", pm))) {
            // //     llvm::errs() << "Failed to parse parallelize\n";
            // // }

            // // // Data Copy & Unroll
            // // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineDataCopyGenerationPass());
            // // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopUnrollPass(4));

            // // // Simplify & Vectorize
            // // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());
            // // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineVectorize());
            // // pm.addPass(createCanonicalizerPass());
            // // pm.addPass(mlir::createLowerAffinePass());

            // // 8. LOWER TO GPU
             // 8. GPU MAPPING
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createGpuMapParallelLoopsPass());
            pm.addPass(mlir::createConvertParallelLoopToGpuPass());
            pm.addPass(mlir::createCanonicalizerPass());
            // 9. GPU KERNEL GENERATION & BINARY COMPILATION
            pm.addPass(mlir::createGpuDecomposeMemrefsPass());
            pm.addPass(mlir::createGpuKernelOutliningPass());

            mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
            nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
            nvvmTargetOptions.chip = "sm_70";
            nvvmTargetOptions.features = "+ptx70";
            pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

            // Lowering INSIDE the GPU Module (Fixes 'index' in kernels)
            auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
            gpuPm.addPass(mlir::createLowerAffinePass());
            gpuPm.addPass(mlir::createSCFToControlFlowPass());
            gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps());
            gpuPm.addPass(mlir::createConvertIndexToLLVMPass()); // MUST be here
            gpuPm.addPass(mlir::createArithToLLVMConversionPass());
            gpuPm.addPass(mlir::createConvertMathToLLVMPass());
            gpuPm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // Binary generation (Stage 2)
            mlir::GpuModuleToBinaryPassOptions binaryOptions;
            binaryOptions.toolkitPath = "/usr/local/cuda-12.1"; // Adjust path if needed
            pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));

            // 10. FINAL HOST LOWERING (Fixes gpu.launch_func and index errors)
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            // PRE-STEP: Prepare types so gpu.launch_func operands are LLVM-compatible
            pm.addPass(mlir::createConvertIndexToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            
            // Note: NovaSymbolFixPass is removed as FixGpuLaunchPass handles dealloc_helper

            // Convert Func to LLVM EARLY to ensure atomic conversion of definitions and calls
            pm.addPass(mlir::createConvertFuncToLLVMPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // MAIN LOWERING: gpu.launch_func -> runtime calls
            mlir::GpuToLLVMConversionPassOptions hostOptions;
            hostOptions.hostBarePtrCallConv = true; // Match bare pointer kernels
            hostOptions.kernelBarePtrCallConv = true;
            pm.addPass(mlir::createGpuToLLVMConversionPass(hostOptions));
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            // FINAL CLEANUP: Ensure NO index or high-level types remain
            pm.addPass(mlir::createSCFToControlFlowPass());
            pm.addPass(mlir::createConvertControlFlowToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::memref::createExpandStridedMetadataPass());
            pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass()); // Critical for llvm.store
            pm.addPass(mlir::nova::createFixGpuLaunchPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            
            
        }

        void registerNovaGPUPipelines()
        {
            PassPipelineRegistration<>("nova-gpu-pipeline",
                                       "Nova GPU Pipeline",
                                       createNovaGPUPipelines);
        }

    } // namespace nova
} // namespace mlir
