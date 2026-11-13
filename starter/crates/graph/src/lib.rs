//! The declarative graph: ops, fusion, backend assignment, allocation, the KV cache.
//!
//! Filled by Chapters 14-19. This crate exists, compiles and has no public surface yet: the book adds
//! one function per lesson, and this placeholder is what makes the layering real before the first
//! line of it is written.

/// The layer this crate is responsible for, for the day a build error is easier to read than a plan.
pub const LAYER: &str = "myfer-graph";

#[cfg(test)]
mod tests {
    #[test]
    fn the_layer_names_itself() {
        assert_eq!(super::LAYER, "myfer-graph");
    }
}
