#include "vector_add.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__        \
                      << ": " << cudaGetErrorString(error) << '\n';             \
            status = 1;                                                         \
            goto cleanup;                                                       \
        }                                                                       \
    } while (0)

__global__ void vector_add_kernel(const float* left, const float* right,
                                  float* output, std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

int main(int argc, char** argv) {
    const std::size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1U << 20);
    if (count == 0) {
        std::cerr << "Vector length must be positive.\n";
        return 2;
    }

    std::vector<float> left(count, 1.25F);
    std::vector<float> right(count, 2.5F);
    std::vector<float> cpu_output(count);
    std::vector<float> gpu_output(count);
    const auto cpu_start = std::chrono::steady_clock::now();
    vector_add_cpu(left.data(), right.data(), cpu_output.data(), count);
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_start).count();

    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_output = nullptr;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    bool kernel_start_created = false;
    bool kernel_stop_created = false;
    int status = 0;
    float kernel_ms = 0.0F;
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    const auto total_start = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventCreate(&kernel_start));
    kernel_start_created = true;
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    kernel_stop_created = true;

    CUDA_CHECK(cudaMalloc(&device_left, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_right, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_left, left.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_right, right.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(kernel_start));
    vector_add_kernel<<<blocks, threads>>>(device_left, device_right, device_output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t index = 0; index < count; ++index) {
        if (std::abs(gpu_output[index] - cpu_output[index]) > 1e-6F) {
            std::cerr << "Validation failed at element " << index << '\n';
            status = 1;
            goto cleanup;
        }
    }

cleanup:
    if (device_output != nullptr && cudaFree(device_output) != cudaSuccess) {
        std::cerr << "CUDA cleanup failed for output buffer.\n";
        status = 1;
    }
    if (device_right != nullptr && cudaFree(device_right) != cudaSuccess) {
        std::cerr << "CUDA cleanup failed for right input buffer.\n";
        status = 1;
    }
    if (device_left != nullptr && cudaFree(device_left) != cudaSuccess) {
        std::cerr << "CUDA cleanup failed for left input buffer.\n";
        status = 1;
    }
    if (kernel_stop_created && cudaEventDestroy(kernel_stop) != cudaSuccess) {
        std::cerr << "CUDA cleanup failed for stop event.\n";
        status = 1;
    }
    if (kernel_start_created && cudaEventDestroy(kernel_start) != cudaSuccess) {
        std::cerr << "CUDA cleanup failed for start event.\n";
        status = 1;
    }
    const double end_to_end_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_start).count();
    if (status != 0) {
        return status;
    }
    std::cout << "Vector-add correctness: PASS (" << count << " elements)\n"
              << "CPU host elapsed: " << cpu_ms << " ms\n"
              << "CUDA kernel event time: " << kernel_ms << " ms\n"
              << "CUDA end-to-end (allocations, H2D, kernel, sync, D2H, cleanup): "
              << end_to_end_ms << " ms\n";
    return 0;
}