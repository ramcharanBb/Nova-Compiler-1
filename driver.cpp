#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <algorithm>

struct MemRefDescriptor {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

extern "C" MemRefDescriptor main1();

int main() {
    const int iterations = 10;
    const int64_t N = 2048;
    // FLOPs: 2*N^3 (matmul) + N^2 (add)
    const double total_flops = (2.0 * N * N * N) + (N * N);
    
    std::cout << "Benchmarking main1 (2048x2048 matrix operations)..." << std::endl;
    
    // Warmup
    MemRefDescriptor res_warm = main1();
    if (res_warm.allocated) free(res_warm.allocated);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        MemRefDescriptor res = main1();
        if (res.allocated) free(res.allocated);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / iterations;
    double gflops = (total_flops / 1e9) / avg_time;
    
    // Theoretical Specs (RTX 3060)
    // Ref: ~3584 cores * 2 * 1.78GHz = 12.7 TFLOPS
    const double theoretical_peak_gflops = 12700.0; 
    // Ref: 192-bit * 15Gbps / 8 = 360 GB/s
    const double theoretical_bandwidth_gbps = 360.0; 
    
    // Memory Intensity calculation
    // Inputs: 3 * 2048*2048 * 4 bytes
    // Output: 1 * 2048*2048 * 4 bytes
    // Total Bytes = 4 * 2048*2048 * 4 = 67,108,864 bytes (~64 MiB)
    const double total_bytes = 4.0 * N * N * sizeof(float);
    double intensity = total_flops / total_bytes;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Workload:              2048x2048 (Matmul + Add)" << std::endl;
    std::cout << "Total Operations:      " << total_flops / 1e9 << " GFLOP" << std::endl;
    std::cout << "Average Execution Time: " << avg_time * 1000.0 << " ms" << std::endl;
    std::cout << "Achieved Performance:  " << gflops << " GFLOPS" << std::endl;
    std::cout << "Theoretical Peak:      " << theoretical_peak_gflops << " GFLOPS" << std::endl;
    std::cout << "Utilization:           " << (gflops / theoretical_peak_gflops) * 100.0 << " %" << std::endl;
    
    std::cout << "\n=== Roofline Analysis ===" << std::endl;
    std::cout << "Arithmetic Intensity:  " << intensity << " FLOP/Byte" << std::endl;
    
    double bandwidth_limited_flops = theoretical_bandwidth_gbps * intensity;
    std::cout << "Bandwidth Bound Limit: " << bandwidth_limited_flops << " GFLOPS" << std::endl;
    
    std::cout << "Performance Limit:     " << std::min(theoretical_peak_gflops, bandwidth_limited_flops) << " GFLOPS" << std::endl;

    if (gflops < bandwidth_limited_flops * 0.5) {
        std::cout << "Status:                MEMORY BOUND (or Overhead Limited)" << std::endl;
    } else if (gflops < theoretical_peak_gflops * 0.5) {
        std::cout << "Status:                COMPUTE BOUND (Sub-optimal)" << std::endl;
    } else {
        std::cout << "Status:                COMPUTE BOUND (Efficient)" << std::endl;
    }
    
    return 0;
}
