//! Tests for the `Box` module.
//!
//! Kept in a separate file so the module root stays focused on the type and
//! its constructors; see [`super`] for the main definitions.

#![allow(unused_imports, clippy::vec_init_then_push)]

extern crate std;
use crate::alloc::CountingAllocator;
use core::any::Any;
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::error::Error;
use core::ffi::CStr;
use core::fmt;
use core::marker::PhantomPinned;
use olive_core::TryClone;

use super::*;
use std::format;
use std::string::String;
use std::vec::Vec;

// Local `vec!` shim: std's `vec!` macro isn't in scope without a prelude,
// and importing it would collide with the `std::vec` module name.
#[allow(clippy::vec_init_then_push)]
macro_rules! vec {
    ($($x:expr),* $(,)?) => {{
        let mut v = <std::vec::Vec<_>>::new();
        $(v.push($x);)*
        v
    }};
}

#[test]
fn test_box_try_new_and_deref() {
    let b = Box::try_new(42i32).unwrap();
    assert_eq!(*b, 42);
}

#[test]
fn test_box_try_new_ok() {
    let b = Box::try_new(99u64).unwrap();
    assert_eq!(*b, 99);
}

#[test]
fn test_box_zst() {
    struct Zst;
    let b = Box::try_new(Zst).unwrap();
    // Should not allocate; should work fine.
    let _ = &b;
}

#[test]
fn test_box_into_raw_from_raw_roundtrip() {
    let b = Box::try_new(String::from("hello")).unwrap();
    let raw: *mut String = Box::into_raw(b);
    assert_eq!(unsafe { &*raw }, "hello");
    // Reconstruct from the raw pointer using an explicit type ascription
    // so inference can resolve both `T` and `A`.
    let b: Box<String, Global> = unsafe { Box::from_raw(raw) };
    assert_eq!(&**b, "hello");
}

#[test]
fn test_box_leak() {
    let b = Box::try_new(vec![1, 2, 3]).unwrap();
    let leaked: &'static mut Vec<i32> = Box::leak(b);
    assert_eq!(leaked, &vec![1, 2, 3]);
}

#[test]
fn test_box_try_clone() {
    let b1 = Box::try_new(5u32).unwrap();
    let b2: Box<u32> = b1.try_clone().unwrap();
    assert_eq!(*b1, *b2);
    assert_ne!(Box::as_ptr(&b1), Box::as_ptr(&b2));
}

#[test]
fn test_box_debug_display() {
    let b = Box::try_new(format_test_string()).unwrap();
    assert_eq!(format!("{b:?}"), "\"hi\"");
    assert_eq!(format!("{b}"), "hi");
}

fn format_test_string() -> String {
    String::from("hi")
}

#[test]
fn test_box_equality_ordering_hash() {
    let a = Box::try_new(10i32).unwrap();
    let b = Box::try_new(10i32).unwrap();
    let c = Box::try_new(20i32).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(a < c);
    assert_eq!(a.cmp(&b), Ordering::Equal);

    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(a);
    assert!(set.contains(&10i32));
}

#[test]
fn test_box_asref_asmut_borrow() {
    let mut b = Box::try_new(7u16).unwrap();
    let r: &u16 = b.as_ref();
    assert_eq!(*r, 7);
    let m: &mut u16 = b.as_mut();
    *m = 8;
    assert_eq!(*b, 8);
    let br: &u16 = b.borrow();
    assert_eq!(*br, 8);
    let bm: &mut u16 = b.borrow_mut();
    *bm = 9;
    assert_eq!(*b, 9);
}

#[test]
fn test_box_try_pin() {
    let p: Pin<Box<u8, Global>> = Box::<u8, Global>::try_pin_in(31u8, Global).unwrap();
    assert_eq!(*p, 31);
}

#[test]
fn test_box_try_pin_global() {
    let p: Pin<Box<u8>> = Box::try_pin(31u8).unwrap();
    assert_eq!(*p, 31);
}

#[test]
fn test_box_try_new_give_back_success() {
    let b = Box::try_new_give_back(42u32).unwrap();
    assert_eq!(*b, 42);
}

#[test]
fn test_box_try_pin_give_back_success() {
    let p: Pin<Box<u32>> = Box::try_pin_give_back(7u32).unwrap();
    assert_eq!(*p, 7);
}

#[test]
fn test_box_try_new_give_back_in_success() {
    let b = Box::<u32, Global>::try_new_give_back_in(42u32, Global).unwrap();
    assert_eq!(*b, 42);
}

#[test]
fn test_box_try_pin_give_back_in_success() {
    let p: Pin<Box<u32, Global>> = Box::<u32, Global>::try_pin_give_back_in(7u32, Global).unwrap();
    assert_eq!(*p, 7);
}

#[test]
fn test_box_uninit_write_and_assume_init() {
    // write(): initialize via the consuming helper.
    let b: Box<MaybeUninit<u64>> = Box::try_new_uninit().unwrap();
    let init = b.write(99u64);
    assert_eq!(*init, 99);

    // assume_init(): manually fill then reinterpret.
    let mut u: Box<MaybeUninit<u64>> = Box::try_new_uninit().unwrap();
    unsafe { (*u).as_mut_ptr().cast::<u64>().write(1234u64) };
    let init = unsafe { u.assume_init() };
    assert_eq!(*init, 1234);
}

#[test]
fn test_box_slice_try_from_slice() {
    let src = [1, 2, 3, 4, 5];
    let bs: Box<[i32]> = Box::try_from_slice(&src).unwrap();
    assert_eq!(bs.len(), 5);
    assert_eq!(bs[0], 1);
    assert_eq!(bs[4], 5);
}

#[test]
fn test_box_slice_try_from_slice_chars() {
    let src = ['a', 'b', 'c'];
    let bs = Box::try_from_slice(&src).unwrap();
    assert_eq!(bs.len(), 3);
    assert_eq!(bs[1], 'b');
}

#[test]
fn test_box_slice_try_clone() {
    let orig: Box<[i32]> = Box::try_from_slice(&[1, 2, 3]).unwrap();
    let cloned: Box<[i32]> = orig.try_clone().unwrap();
    assert_eq!(orig, cloned);
    assert_ne!(Box::as_ptr(&orig), Box::as_ptr(&cloned));
}

#[test]
fn test_box_str_try_from_ref() {
    let bs: Box<str> = Box::try_clone_from_ref("hello world").unwrap();
    assert_eq!(&*bs, "hello world");
    assert_eq!(bs.len(), 11);
}

#[test]
fn test_box_str_try_clone() {
    let orig: Box<str> = Box::try_clone_from_ref("abc").unwrap();
    let cloned: Box<str> = orig.try_clone().unwrap();
    assert_eq!(orig, cloned);
}

#[test]
fn test_box_cstr_try_clone() {
    // Build a `Box<CStr>` by cloning from a `&CStr` reference — this exercises
    // both the allocator-clone seam and the `TryCloneToUninit for CStr` impl.
    let src = c"hello";
    let boxed: Box<CStr, Global> = Box::try_clone_from_ref(src).unwrap();
    assert_eq!(boxed.to_bytes_with_nul(), b"hello\x00");

    let cloned: Box<CStr, Global> = boxed.try_clone().unwrap();
    assert_eq!(cloned.to_bytes_with_nul(), b"hello\x00");
    assert_ne!(Box::as_ptr(&boxed), Box::as_ptr(&cloned));
}

#[test]
fn test_box_str_to_bytes() {
    let bs: Box<str> = Box::try_clone_from_ref("xyz").unwrap();
    let bytes: Box<[u8]> = Box::from(bs);
    assert_eq!(&bytes[..], b"xyz");
}

#[test]
fn test_box_dyn_trait_object() {
    trait Greet {
        fn greet(&self) -> String;
    }
    // Non-zero-sized payload so the boxed trait object owns real data on
    // the heap (the vtable itself lives in the fat pointer's metadata).
    #[allow(dead_code)]
    struct Dog {
        tag: u64,
    }
    impl Greet for Dog {
        fn greet(&self) -> String {
            String::from("woof")
        }
    }
    // Box the concrete type, then unsize-coerce to a trait object through a
    // raw fat pointer. The concrete allocation carries the payload and the
    // coercion attaches the vtable — the sound way to get a `Box<dyn Trait>`.
    let boxed: Box<Dog, Global> = Box::try_new(Dog { tag: 1 }).unwrap();
    let raw: *mut Dog = Box::into_raw(boxed);
    let fat: *mut dyn Greet = raw as *mut dyn Greet;
    let b: Box<dyn Greet, Global> = unsafe { Box::from_raw_in(fat, Global) };
    assert_eq!(b.greet(), "woof");
}

#[test]
fn test_box_dyn_zst_concrete() {
    trait Greet {
        fn greet(&self) -> String;
    }
    struct Empty; // ZST
    impl Greet for Empty {
        fn greet(&self) -> String {
            String::from("...")
        }
    }
    // A ZST behind a trait object has no data bytes on the heap; only the
    // vtable (in the fat-pointer metadata) distinguishes it. Boxing the
    // concrete value and unsize-coercing through a raw fat pointer yields a
    // working `Box<dyn Greet>` whose methods dispatch via the vtable.
    let boxed: Box<Empty, Global> = Box::try_new(Empty).unwrap();
    let raw: *mut Empty = Box::into_raw(boxed);
    let fat: *mut dyn Greet = raw as *mut dyn Greet;
    let b: Box<dyn Greet, Global> = unsafe { Box::from_raw_in(fat, Global) };
    assert_eq!(b.greet(), "...");
}

#[test]
fn test_box_any_downcast() {
    // Build a `Box<dyn Any>` by boxing the concrete value and unsize-coercing
    // through a raw fat pointer (our `Box::try_new` is sized-only).
    let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
    let raw: *mut i32 = Box::into_raw(boxed_i32);
    let fat: *mut dyn Any = raw as *mut dyn Any;
    let b: Box<dyn Any, Global> = unsafe { Box::from_raw_in(fat, Global) };
    let recovered = b.downcast::<i32>().unwrap();
    assert_eq!(*recovered, 42);

    let boxed_i32: Box<i32, Global> = Box::try_new_in(42i32, Global).unwrap();
    let raw: *mut i32 = Box::into_raw(boxed_i32);
    let fat: *mut dyn Any = raw as *mut dyn Any;
    let b: Box<dyn Any, Global> = unsafe { Box::from_raw_in(fat, Global) };
    assert!(b.downcast::<f64>().is_err());
}

#[test]
fn test_box_error_downcast() {
    #[derive(Debug)]
    struct MyErr;
    impl fmt::Display for MyErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "my error")
        }
    }
    impl Error for MyErr {}

    // Box the concrete error and unsize-coerce through a raw fat pointer.
    let boxed: Box<MyErr, Global> = Box::try_new(MyErr).unwrap();
    let raw: *mut MyErr = Box::into_raw(boxed);
    let fat: *mut dyn Error = raw as *mut dyn Error;
    let e: Box<dyn Error, Global> = unsafe { Box::from_raw_in(fat, Global) };
    let recovered = e.downcast::<MyErr>().unwrap();
    let _ = recovered; // just verify it works
}

#[test]
fn test_box_iterator_forwarding() {
    let v = vec![1, 2, 3, 4, 5];
    let iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
    let collected: Vec<&i32> = iter.collect();
    assert_eq!(collected, vec![&1, &2, &3, &4, &5]);
}

#[test]
fn test_box_double_ended_iterator() {
    let v = vec![1, 2, 3, 4, 5];
    let mut iter: Box<std::slice::Iter<'_, i32>> = Box::try_new(v.iter()).unwrap();
    assert_eq!(iter.next(), Some(&1));
    assert_eq!(iter.next_back(), Some(&5));
    assert_eq!(iter.next(), Some(&2));
}

#[test]
fn test_box_custom_allocator_global() {
    // Verify that Box works with an explicit `Global` allocator; the
    // stored allocator must be the same unit struct we passed in.
    let b: Box<u32, Global> = Box::try_new_in(77u32, Global).unwrap();
    assert_eq!(*b, 77);
    assert!(matches!(b.allocator(), Global));
}

#[test]
fn test_box_try_new_in() {
    let b = Box::try_new_in(55u16, Global).unwrap();
    assert_eq!(*b, 55);
}

#[test]
fn test_box_slice_try_from() {
    let bs: Box<[i32]> = Box::try_from_slice(&[1, 2, 3]).unwrap();
    let arr: Box<[i32; 3]> = Box::try_from(bs).unwrap();
    assert_eq!(arr[0], 1);
    assert_eq!(arr[2], 3);

    let bs: Box<[i32]> = Box::try_from_slice(&[1, 2]).unwrap();
    let result: Result<Box<[i32; 3]>, Box<[i32]>> = Box::try_from(bs);
    assert!(result.is_err());
}

#[test]
fn test_box_pointer_fmt() {
    let b = Box::try_new(0u8).unwrap();
    let _ = format!("{b:p}");
}

// -----------------------------------------------------------------------
// Free-on-panic regression tests
//
// Dropping a `Box` whose pointee's destructor panics must still release the
// backing allocation (otherwise the box leaks on unwind). This holds for a
// plain `Box` and for a `Pin<Box>` alike — pinning only forbids *moving* the
// pointee, and neither dropping it in place nor freeing its block relocates
// it, so the two behave identically here. We observe the free through a
// counting `StaticAllocator` rather than relying on Miri's leak heuristic.
// -----------------------------------------------------------------------

/// Explicitly NOT `Unpin` (via `PhantomPinned`) AND non-zero-sized (via the
/// `u32` payload), so `Pin<Box<_>>` is a genuinely pinned box over real heap
/// memory. Its destructor panics unconditionally.
struct PanicOnDrop {
    _payload: u32,
    _pin: PhantomPinned,
}

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        panic!("forced panic in pointee drop");
    }
}

#[test]
fn test_box_plain_panic_on_drop_frees_allocation() {
    // Pass a *reference* to the allocator so the box stores a `&CountingAllocator`;
    // we keep the original handle to read the counters after the (panicking) drop.
    let alloc = CountingAllocator::new();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let b: Box<PanicOnDrop, &CountingAllocator> = Box::try_new_in(
            PanicOnDrop {
                _payload: 0,
                _pin: PhantomPinned,
            },
            &alloc,
        )
        .unwrap();
        drop(b);
    }));
    assert!(res.is_err(), "expected the pointee drop to panic");
    assert_eq!(alloc.allocations(), 1, "expected one allocation");
    assert_eq!(
        alloc.deallocations(),
        1,
        "plain Box must free its allocation even when the pointee's drop panics"
    );
}

#[test]
fn test_box_pinned_panic_on_drop_frees_allocation() {
    let alloc = CountingAllocator::new();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let p: Pin<Box<PanicOnDrop, &CountingAllocator>> = Box::try_pin_in(
            PanicOnDrop {
                _payload: 0,
                _pin: PhantomPinned,
            },
            &alloc,
        )
        .unwrap();
        drop(p);
    }));
    assert!(res.is_err(), "expected the pointee drop to panic");
    assert_eq!(alloc.allocations(), 1, "expected one allocation");
    assert_eq!(
        alloc.deallocations(),
        1,
        "Pin<Box> must free its allocation even when the pointee's drop panics"
    );
}
