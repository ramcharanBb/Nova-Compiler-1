// convert to llvm includes
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// other conversion includes from mlir
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
// utils
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h" // Correct header for Linalg passes even if Affine conversion is hidden
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

    pm.addNestedPass<func::FuncOp>(mlir::nova::createFuseMatmulBiasPass());
    pm.addPass(createCanonicalizerPass());

    bufferization::OneShotBufferizePassOptions bufferizeOptions;
    bufferizeOptions.bufferizeFunctionBoundaries = true;         
    bufferizeOptions.functionBoundaryTypeConversion =            
        bufferization::LayoutMapOption::IdentityLayoutMap;


    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));

    bufferization::BufferDeallocationPipelineOptions deallocationOptions;
    bufferization::buildBufferDeallocationPipeline(pm, deallocationOptions);

    //   Lower Linalg to Affine loops
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    
    // Tile Affine loops
    if (failed(parsePassPipeline("func.func(affine-loop-tile{tile-sizes=32,32,32})", pm)))
    {
        llvm::errs() << "Failed to parse affine-loop-tile pipeline\n";
    }
    parsePassPipeline("func.func(affine-parallelize)", pm) ;
    //   // Lower Affine to SCF to continue the pipeline
    pm.addPass(mlir::createLowerAffinePass());
}
