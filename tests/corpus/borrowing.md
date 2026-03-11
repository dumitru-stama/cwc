# Borrowing in Rust

## References and Borrowing

A reference allows you to refer to a value without taking ownership of it. The action of creating a reference is called borrowing. References are immutable by default.

```rust
fn calculate_length(s: &String) -> usize {
    s.len()
}

let s1 = String::from("hello");
let len = calculate_length(&s1);
// s1 is still valid here
```

## Mutable References

You can create a mutable reference with `&mut`. Mutable references allow you to modify the borrowed value. However, you can have only one mutable reference to a particular piece of data at a time. This restriction prevents data races at compile time.

```rust
let mut s = String::from("hello");
let r1 = &mut s;
r1.push_str(" world");
```

## The Rules of References

Rust enforces two rules for references:
1. At any given time, you can have either one mutable reference or any number of immutable references.
2. References must always be valid (no dangling references).

These rules are enforced at compile time by the borrow checker.

## Dangling References

Rust prevents dangling references at compile time. A dangling reference is a pointer that references a location in memory that may have been given to someone else. The compiler ensures that data will not go out of scope before the reference to the data does.

## Slices

Slices let you reference a contiguous sequence of elements rather than the whole collection. A string slice is a reference to part of a String. Slices are a kind of reference, so they do not have ownership.

```rust
let s = String::from("hello world");
let hello = &s[0..5];
let world = &s[6..11];
```
