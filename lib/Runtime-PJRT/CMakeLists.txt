# Nova Runtime Library

add_library(NovaRuntime
  NovaDevice.cpp
  NovaBuffer.cpp
  NovaClient.cpp
  NovaExecutable.cpp
)

target_include_directories(NovaRuntime
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(NovaRuntime
  PUBLIC
    MLIRIR
    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
)

# Optional: CUDA support
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  message(STATUS "CUDA found - enabling CUDA support in NovaRuntime")
  target_compile_definitions(NovaRuntime PRIVATE NOVA_ENABLE_CUDA)
  target_link_libraries(NovaRuntime PRIVATE CUDA::cuda_driver)
  target_include_directories(NovaRuntime PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else()
  message(STATUS "CUDA not found - NovaRuntime will only support CPU")
endif()

# Install
install(TARGETS NovaRuntime
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)