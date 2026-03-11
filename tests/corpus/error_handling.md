# Error Handling in Rust

## Unrecoverable Errors with panic!

When the `panic!` macro executes, your program will print a failure message, unwind and clean up the stack, and then quit. You can set the `RUST_BACKTRACE` environment variable to get a backtrace of what happened.

## Recoverable Errors with Result

The `Result` enum has two variants: `Ok(T)` for success and `Err(E)` for errors. Most errors are recoverable and should use Result rather than panicking.

```rust
use std::fs::File;

fn read_file() -> Result<String, std::io::Error> {
    let f = File::open("hello.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}
```

## The ? Operator

The `?` operator can be used with functions that return `Result`. If the value is `Ok`, it unwraps the value. If the value is `Err`, the error is returned from the whole function. This makes error propagation concise.

## Custom Error Types

You can define custom error types using enums. The `thiserror` crate makes this easier with derive macros. Custom error types should implement `std::error::Error`, `Display`, and `Debug`.

```rust
#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),
}
```

## When to Panic

Use `panic!` when your code is in an unrecoverable state. Use `Result` when failure is an expected possibility. In tests, `unwrap` and `expect` are appropriate because a failing test should panic.

## The anyhow and thiserror Crates

The `anyhow` crate provides a convenient error type for application code. The `thiserror` crate helps define custom error types for library code with derive macros.
