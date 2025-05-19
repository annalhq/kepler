// use of null pointer

/* remember:
1. Initializing a pointer to NULL is a good practice to avoid dangling pointers.
2. Check pointers for NULL before dereferencing them to prevent segmentation faults.
3. NULL checks allows handling of errors gracefully.
4. NULL pointers can be used to indicate the absence of a value or an uninitialized state.
*/

#include <stdio.h>
#include <stdlib.h>

int main() {
    int* ptr = NULL; // Initialize pointer to NULL
    printf("1. Initial pointer value: %p\n", (void*)ptr);

    // check for null before using it
    if (ptr == NULL) printf("ptr cannot be dereferenced");

    // allocate mem
    ptr = malloc(sizeof(int));
    if (ptr == NULL)
    {
        printf("3. Memory allocation failed");
        return 1;
    }
    
    printf("4. After allocation, ptr value: %p\n", (void*)ptr);

    // Safe to use ptr after NULL check
    *ptr = 42;
    printf("5. Value at ptr: %d\n", *ptr);

    // Clean up
    free(ptr);
    ptr = NULL;  // Set to NULL after freeing

    printf("6. After free, ptr value: %p\n", (void*)ptr);

    // Demonstrate safety of NULL check after free
    if (ptr == NULL) {
        printf("7. ptr is NULL, safely avoided use after free\n");
    }

    return 0;
}