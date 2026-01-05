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
#include "Compiler/Transforms/Passes.h"
#include "Compiler/Transforms/AddGpuMemoryCopies.h"

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
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"


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
        namespace
        {
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
            // This enables the 2:4 structured sparsity hardware path on your RTX 3060.
            // pm.addPass(mlir::createSparsificationPass());
            // pm.addPass(mlir::createSparseTensorConversionPass());

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
            
            //  TILING & VECTORIZATION
            // pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToGPUPass(true));
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nvgpu::createOptimizeSharedMemoryPass());
            pm.addPass(mlir::createCanonicalizerPass());

            //5. BUFFERIZATION & DEALLOCATION
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
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
            // Apply Tiling HERE on the parallel loops
            pm.addPass(mlir::createParallelLoopTilingPass({16, 16, 16}));
            pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopFusionPass());
            pm.addPass(mlir::createCanonicalizerPass());
            //8. GPU MAPPINGcreateParallelLoopFusionPass
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createGpuMapParallelLoopsPass());
            pm.addPass(mlir::createConvertParallelLoopToGpuPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createGpuKernelOutliningPass());
            
            // Add custom memory management pass
            pm.addNestedPass<mlir::func::FuncOp>(mlir::nova::createAddGpuMemoryCopiesPass());

            mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
            nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
            nvvmTargetOptions.chip = "sm_86";    
            pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

            // Lowering INSIDE the GPU Module (Fixes 'index' in kernels)
            auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
            gpuPm.addPass(mlir::createLowerAffinePass());
            gpuPm.addPass(mlir::createSCFToControlFlowPass());
            mlir::ConvertGpuOpsToNVVMOpsOptions nvvmOptions;
            gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvmOptions));
            gpuPm.addPass(mlir::createConvertIndexToLLVMPass());
            gpuPm.addPass(mlir::createArithToLLVMConversionPass());
            gpuPm.addPass(mlir::createConvertMathToLLVMPass());
            gpuPm.addPass(mlir::createReconcileUnrealizedCastsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // Binary generation (Stage 2)
            mlir::GpuModuleToBinaryPassOptions binaryOptions;
            binaryOptions.toolkitPath = "/usr/local/cuda-13.0";
            pm.addPass(mlir::createGpuModuleToBinaryPass(binaryOptions));

            // 10. FINAL HOST LOWERING (Fixes gpu.launch_func and index errors)
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createConvertIndexToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());

            // MAIN LOWERING: gpu.launch_func -> runtime calls
            mlir::GpuToLLVMConversionPassOptions hostOptions;
            pm.addPass(mlir::createGpuToLLVMConversionPass(hostOptions));
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createSCFToControlFlowPass());
            pm.addPass(mlir::createConvertControlFlowToLLVMPass());
            pm.addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(mlir::memref::createExpandStridedMetadataPass());
            pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass()); 
            pm.addPass(mlir::nova::createFixGpuLaunchPass()); 
            pm.addPass(mlir::createConvertFuncToLLVMPass()); 
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
