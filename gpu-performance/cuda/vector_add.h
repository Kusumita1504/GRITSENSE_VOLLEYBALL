#ifndef GRITSENSE_VECTOR_ADD_H
#define GRITSENSE_VECTOR_ADD_H

#include <cstddef>

void vector_add_cpu(const float* left, const float* right, float* output, std::size_t count);

#endif