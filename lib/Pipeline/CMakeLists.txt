add_mlir_library(NovaPipelines
  Pipeline.cpp
  Gpupipeline.cpp 
     

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Compiler

  DEPENDS
  MLIRNovaOpsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRAffineTransformPasses 
  AffineTransforms 
  MLIRAffineTransforms
  
  MLIRNova
  
  MLIRNovaToArithTransforms
  MLIRNovaToLinalgTransforms
  MLIRNovaToTosaTransforms

  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRBufferizationToMemRef
  MLIRGPUTransforms
  MLIRGPUToNVVMTransforms
  MLIRNVVMToLLVMIRTranslation
)

target_include_directories(NovaPipelines PUBLIC 
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/Compiler/Pipeline>
)