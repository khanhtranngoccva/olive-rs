use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// A value type whose `TryDefault` always fails, used to exercise the
/// default-construction-failure path (e.g. `Entry::or_try_default`).
#[derive(Debug)]
pub struct NoDefault;

impl TryDefault for NoDefault {
    fn try_default() -> Result<Self, TryDefaultError> {
        Err(TryDefaultError::Other("no canonical value"))
    }
}
