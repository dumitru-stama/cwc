# Lifetimes in Rust

## What are Lifetimes?

Every reference in Rust has a lifetime, which is the scope for which that reference is valid. Most of the time, lifetimes are implicit and inferred. We must annotate lifetimes when the lifetimes of references could be related in a few different ways.

## Lifetime Annotation Syntax

Lifetime annotations describe the relationships of the lifetimes of multiple references to each other. They use an apostrophe followed by a short lowercase name, usually starting with `'a`.

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

## Lifetime Elision Rules

The Rust compiler has three lifetime elision rules that allow you to omit lifetime annotations in common cases:
1. Each parameter that is a reference gets its own lifetime parameter.
2. If there is exactly one input lifetime parameter, that lifetime is assigned to all output lifetime parameters.
3. If one of the input parameters is `&self` or `&mut self`, the lifetime of self is assigned to all output lifetime parameters.

## The Static Lifetime

The `'static` lifetime denotes that the affected reference can live for the entire duration of the program. All string literals have the `'static` lifetime because they are stored directly in the program binary.

```rust
let s: &'static str = "I have a static lifetime.";
```

## Lifetimes in Structs

When a struct holds references, it needs lifetime annotations. This tells Rust that an instance of the struct cannot outlive the reference it holds.

```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}
```
