//! Build-time environment detection and other utilities for the Olive workspace.
//!
//! This crate is consumed exclusively as a `build-dependency`. It provides
//! granular predicates so each consuming crate's build script can decide
//! which cfg flags to emit.
//!
//! # Detection items
//!
//! ## Unstable features
//!
//! Two independent signals indicate that unstable features (`#![feature(...)]`)
//! are permitted:
//!
//! - **Nightly** — the compiler's version string contains `"nightly"` or
//!   `"-dev"`. This is authoritative; no environment variable is involved.
//! - **Bootstrap** — the `RUSTC_BOOTSTRAP` environment variable is set and
//!   applies to the current crate. This allows a stable or beta compiler to
//!   accept `#![feature(...)]` gates (used by the rustc bootstrap process and
//!   by projects like Firefox/Chromium that need cutting-edge features on
//!   stable).
//!
//! A crate may use unstable features when *either* signal is present. The
//! combined result maps to `cfg(unstable_features)`, emitted by the
//! consuming build script.
//!
//! ## Atomic load/store probing
//!
//! The probe asks the compiler for the precise (unstable)
//! `target_has_atomic_load_store` cfg family on the target, and
//! the module's emit helpers forward the confirmed widths as cfg flags using an
//! altered name (to avoid conflict) and register them with `cargo:rustc-check-cfg`.
//!
//! # Cargo build-script directives
//!
//! Directives (cfg emission, check-cfg registration, rerun triggers) are
//! emitted with the `cargo_build` crate.
//!
//! # Re-run triggers
//!
//! Every check emits its own `cargo:rerun-if-env-changed` directives for
//! the environment variables it reads. Build scripts therefore never need
//! to emit rerun triggers by hand, and cached results cannot go stale when
//! a variable changes. cargo treats repeated identical directives as a single
//! trigger.
//!
//! # Rationales
//!
//! ## Running Miri with full provenance
//!
//! There are certain parts where Miri cannot successfully run without unstable features:
//! - Cloning unsized heap items from references require copying pointer metadata. Normally,
//!   we can simply use a pointer hack, but it causes undefined behavior when on Miri
//!   (although it is sound otherwise).
//!
//! Consuming crates therefore refuse to build under a *genuine* Miri invocation
//! unless `unstable_features` is active. The guard is
//! [`miri::ensure_miri_on_unstable`], which returns an error for the build
//! script to surface rather than exiting or panicking.
//!
//! ## Atomic pointer gating
//!
//! - Some targets do not fully support all atomic operations, but they support
//!   some of them. This is necessary for gating items that require atomic pointers.
pub mod miri;
pub mod target_has_atomic_load_store;
pub mod unstable_features;
