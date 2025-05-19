#include <stdio.h>

int main() {
    int num = 10;
    float fnum = 3.1415926;
    void* vptr;

    vptr = &num;

    printf("integer: %d\n", *(int*)vptr);

    vptr = &fnum;
    printf("float integer: %.7f\n", *(float*)vptr);
}
// void pointers are used when we don't know the data type of the memory address
// NOTE: malloc() returns a void pointer but we see it as a pointer to a specific data type after the cast (int*)malloc(4) or (float*)malloc(4) etc.// void pointers are used when we don't know the data type of the memory address
// fun fact: malloc() returns a void pointer but we see it as a pointer to a specific data type after the cast (int*)malloc(4) or (float*)malloc(4) etc