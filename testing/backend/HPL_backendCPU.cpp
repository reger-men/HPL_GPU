#include "backend/hpl_backendCPU.h"

void CPU::malloc(void** ptr, size_t size)
{
    *ptr = std::malloc(size);
    printf("allocate memory on CPU\n");
}