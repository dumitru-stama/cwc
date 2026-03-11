# Ownership in Rust

## What is Ownership?

Ownership is Rust's most unique feature, and it enables Rust to make memory safety guarantees without needing a garbage collector. In Rust, each value has a variable that is called its owner. There can only be one owner at a time. When the owner goes out of scope, the value will be dropped.

## The Stack and the Heap

Both the stack and the heap are parts of memory available at runtime. The stack stores values in a last-in, first-out order. All data stored on the stack must have a known, fixed size. Data with an unknown size at compile time or a size that might change must be stored on the heap.

## The Move Semantics

When you assign a value to another variable, the ownership moves. This is called a move. After a move, the original variable can no longer be used. This prevents double-free errors at compile time.

```rust
let s1 = String::from("hello");
let s2 = s1; // s1 is moved to s2
// println!("{}", s1); // ERROR: s1 is no longer valid
```

## The Clone Trait

If you want to deeply copy the heap data, you can use the `clone` method. This creates a full copy and both variables remain valid.

```rust
let s1 = String::from("hello");
let s2 = s1.clone();
println!("{} {}", s1, s2); // Both valid
```

## The Copy Trait

Types that have a known size at compile time and are stored entirely on the stack can implement the Copy trait. For these types, assignment creates a copy rather than a move. Integer types, floating-point types, booleans, and characters all implement Copy.

## Ownership and Functions

Passing a value to a function will move or copy, just as assignment does. Returning values from functions can also transfer ownership.
