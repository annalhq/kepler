# Advanced C Macros and Global Variables

## 1. Macro Fundamentals

- Textual substitution happens before the compiler sees your code.  
- `#define NAME` replacement  
  - No type checking, scope-blind.  
- Undefine with `#undef` to avoid name collisions.  

## 2. Parameterized Macros: Power and Pitfalls

```c
#define SQR(x) ((x) * (x))
```
- Always wrap parameters & entire expression in parentheses.  
- Beware of side effects: `SQR(i++)` expands to `((i++) * (i++))`.  
- Use inline functions where type safety or side-effect control is critical.  

## 3. Token Pasting and Stringification

- `##` concatenates tokens:  
  ```c
  #define MAKE_VAR(n) var##n
  ```
- `#` stringifies:  
  ```c
  #define TO_STR(x) #x
  ```
- Combine for logging:  
  ```c
  #define LOG(fmt, ...) printf("[%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
  ```

## 4. Conditional Compilation

- `#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif`  
- Use for configuration flags and feature toggles.  
- Always close your conditionals and prefer named macros over magic numbers.  

## 5. Include Guards vs. #pragma once

- Guards: `#ifndef FOO_H` / `#define FOO_H` ... `#endif` for portability.  
- `#pragma once` is simpler but non-standard (widely supported).  

## 6. Macro “Functions” for Generics

```c
#define MAX(a,b) ((a) > (b) ? (a) : (b))
```
- Works on any type supporting `>`, but beware of multiple evaluations.  
- C11 `_Generic` offers type-safe alternatives.  

## 7. Debug and Trace Macros

```c
#ifdef DEBUG
  #define DPRINT(...) fprintf(stderr, __VA_ARGS__)
#else
  #define DPRINT(...)
#endif
```

## 8. Macro Hygiene & Readability

- Keep macros short and purposeful.  
- Document side effects and expansion behavior.  
- Use ALL_CAPS to distinguish from functions/variables.  

------------------------------------------------------------

## 9. Global Variables: Overview

- Declared at file scope; default linkage is external.  
- Storage duration is the lifetime of the program.  
- Access from other translation units via `extern`.  

  ```c
  // in globals.h
  extern int g_counter;
  // in globals.c
  int g_counter = 0;
  ```

## 10. Linkage and Storage Classes

- `static` at file scope → internal linkage (module-private).  
- `extern` → external linkage.  
- `auto` (default inside functions) → automatic storage.  
- `register` (hint) → potentially stored in CPU register.  
- `thread_local` (C11) → one instance per thread.  

## 11. Initialization Rules

- Globals default to zero if uninitialized.  
- Constant initialization at compile time, dynamic initialization at startup.  
- Watch out for inter-module init order (“static initialization fiasco”).  

## 12. Best Practices for Globals

- Minimize use; wrap access in getter/setter functions.  
- Group related globals in structs to improve locality and namespace.  
- Document thread-safety assumptions or protect with mutexes.  
- Use `const` for read-only globals to catch unintended writes.  

## 13. Pitfalls & Gotchas

- Name collisions → use prefixes: `mylib_g_var`.  
- Hidden dependencies across modules.  
- Harder to test and reason about in multi-threaded contexts.  

------------------------------------------------------------
References:
  • The C Programming Language (K&R), Chapter 8
  • C11 Standard §6.10 (Macros), §6.2.1–6.2.2 (Linkages)
  • “Modern C” by Jens Gustedt