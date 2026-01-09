#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

struct MemRefDescriptor {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

extern "C" void _mlir_ciface_main1(MemRefDescriptor* result);

int main() {
    const int iterations = 10;
    const int64_t N = 64; // Scaled down
    const double total_flops = (2.0 * N * N * N) + (N * N) + (N * N); // Matmul + 2 Adds
    
    std::cout << "Benchmarking main1 (64x64 GPU Chain)..." << std::endl;
    
    MemRefDescriptor resultDesc; 
    
    // Warmup
    _mlir_ciface_main1(&resultDesc);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        _mlir_ciface_main1(&resultDesc);
    }
    cudaDeviceSynchronize(); 
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / iterations;
    double gflops = (total_flops / 1e9) / avg_time;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Workload:              64x64 (Add -> Matmul -> Add)" << std::endl;
    std::cout << "Total Operations:      " << total_flops / 1e9 << " GFLOP" << std::endl;
    std::cout << "Average Execution Time: " << avg_time * 1000.0 << " ms" << std::endl;
    std::cout << "Achieved Performance:  " << gflops << " GFLOPS" << std::endl;
    
    return 0;
}
