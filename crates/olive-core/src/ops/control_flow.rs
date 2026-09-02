//! Foundational-trait impls for [`core::ops::ControlFlow`].
//!
//! [`ControlFlow`] is the enum returned by the loop-control combinators
//! (`break`, `continue`, `?`-style early exits). Its two variants are `Break(B)`
//! and `Continue(C)`, each carrying a payload. Cloning one clones the carried
//! payload via [`TryClone`], so a fallible inner value propagates its error rather
//! than panicking on allocation failure.
use core::ops::ControlFlow;

use crate::try_traits::try_clone::{TryClone, TryCloneError};

/// Fallibly clone a [`ControlFlow`], cloning whichever payload is carried.
impl<B: TryClone, C: TryClone> TryClone for ControlFlow<B, C> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            ControlFlow::Break(value) => Ok(ControlFlow::Break(value.try_clone()?)),
            ControlFlow::Continue(value) => Ok(ControlFlow::Continue(value.try_clone()?)),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn control_flow_break_try_clone() {
        let cf: ControlFlow<u32, ()> = ControlFlow::Break(42u32);
        assert_eq!(cf.try_clone().unwrap(), cf);
    }

    #[test]
    fn control_flow_continue_default_payload() {
        // With the default `C = ()`, `Continue` carries a unit payload.
        let cf: ControlFlow<u8, ()> = ControlFlow::Continue(());
        assert_eq!(cf.try_clone().unwrap(), cf);
    }

    #[test]
    fn control_flow_both_payloads_cloned() {
        // Give both `B` and `C` real payloads to exercise both arms.
        let cf: ControlFlow<u32, u64> = ControlFlow::Continue(99u64);
        let cloned = cf.try_clone().unwrap();
        assert_eq!(cloned, cf);
    }

}
