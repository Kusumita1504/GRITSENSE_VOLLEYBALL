#include "vector_add.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

void vector_add_cpu(const float* left, const float* right, float* output, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = left[index] + right[index];
    }
}

#ifndef VECTOR_ADD_NO_MAIN
int main(int argc, char** argv) {
    const std::size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1U << 20);
    if (count == 0) {
        std::cerr << "Vector length must be positive.\n";
        return 2;
    }

    std::vector<float> left(count, 1.25F);
    std::vector<float> right(count, 2.5F);
    std::vector<float> output(count);
    const auto start = std::chrono::steady_clock::now();
    vector_add_cpu(left.data(), right.data(), output.data(), count);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (output.front() != 3.75F || output.back() != 3.75F) {
        std::cerr << "CPU vector-add validation failed.\n";
        return 1;
    }
    std::cout << "CPU vector add: elements=" << count << " host_elapsed_ms=" << elapsed << '\n';
    return 0;
}
#endif