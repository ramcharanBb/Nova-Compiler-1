
# Script to run robust tests

MLIR_OPT="./build/tools/nova-opt/nova-opt"
MLIR_TRANSLATE="../../llvm-project/build/bin/mlir-translate"
LLC="../../llvm-project/build/bin/llc"
INPUT_FILE="ltest/robust_tests.mlir"

echo "=== Robustness Verification Pipeline ==="

# Step 1: Lowering
echo "Step 1: Lowering MLIR..."
$MLIR_OPT $INPUT_FILE \
  --nova-gpu-pipeline \
  -o intermediate_robust.mlir

if [ $? -ne 0 ]; then echo "Lowering failed"; exit 1; fi

# Step 2: Binary Generation
echo "Step 2: Generating GPU binary..."
$MLIR_OPT intermediate_robust.mlir \
  --gpu-module-to-binary="format=isa" \
  -o with_binary_robust.mlir

if [ $? -ne 0 ]; then echo "Binary generation failed"; exit 1; fi

# Step 3: LLVM IR
echo "Step 3: Translating to LLVM IR..."
$MLIR_TRANSLATE --mlir-to-llvmir with_binary_robust.mlir -o output_robust.ll

if [ $? -ne 0 ]; then echo "Translation failed"; exit 1; fi

# Step 4: Object Code
echo "Step 4: Compiling object code..."
$LLC -filetype=obj -relocation-model=pic output_robust.ll -o output_robust.o
clang++ -c robust_driver.cpp -o robust_driver.o

# Step 5: Linking
echo "Step 5: Linking..."
clang++ robust_driver.o output_robust.o -o robust_tests.out \
  -L/usr/local/cuda/lib64 -lcudart -ldl -lm -lmlir_cuda_runtime \
  -L/home/blu-bridge023/Desktop/llvm-project/build/lib

if [ $? -ne 0 ]; then echo "Linking failed"; exit 1; fi

# Step 6: Execution
echo "Step 6: Executing tests..."
LD_LIBRARY_PATH=/home/blu-bridge023/Desktop/llvm-project/build/lib:$LD_LIBRARY_PATH ./robust_tests.out
