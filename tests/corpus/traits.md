# Traits in Rust

## Defining Traits

A trait defines functionality a particular type has and can share with other types. Traits are similar to interfaces in other languages. You define a trait with the `trait` keyword followed by the trait name and a block of method signatures.

```rust
pub trait Summary {
    fn summarize(&self) -> String;
}
```

## Implementing Traits

To implement a trait on a type, use the `impl Trait for Type` syntax. Each type that implements the trait must provide its own implementation of the methods.

```rust
struct NewsArticle {
    headline: String,
    content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}: {}", self.headline, &self.content[..50])
    }
}
```

## Default Implementations

Traits can provide default method implementations. Types can keep the default or override it.

## Trait Bounds

You can use trait bounds to restrict generic types. The syntax `fn notify(item: &impl Summary)` is syntactic sugar for `fn notify<T: Summary>(item: &T)`. You can require multiple traits with the `+` syntax.

## Dynamic Dispatch

Trait objects allow for dynamic dispatch using `dyn Trait`. When using trait objects, Rust uses the pointers inside the trait object to know which method to call. This lookup incurs a runtime cost compared to static dispatch.

```rust
fn print_summary(item: &dyn Summary) {
    println!("{}", item.summarize());
}
```

## Supertraits

A trait can require another trait to be implemented. The required trait is called a supertrait. For example, `trait OutlinePrint: Display` requires that any type implementing OutlinePrint also implements Display.

## The Orphan Rule

You can only implement a trait on a type if either the trait or the type is local to your crate. This is called the orphan rule and prevents conflicting implementations.
