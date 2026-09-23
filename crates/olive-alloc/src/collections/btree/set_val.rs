/// Zero-Sized Type (ZST) for internal `BTreeSet` values.
/// Used instead of `()` to differentiate between:
/// * `BTreeMap<T, ()>` (possible user-defined map)
/// * `BTreeMap<T, SetValZST>` (internal set representation)
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Default)]
pub(super) struct SetValZST;

// Distinguishing between the above ZST and other types requires specialization,
// which has not landed yet in stable. We can ignore it though as it is only used
// for internal panics.
