//! Newtypes whose trait operations blow up on command via *panics*, for tests that
//! must break an algorithm mid-flight.

pub mod panicky_ord;
pub use panicky_ord::PanickyOrd;
