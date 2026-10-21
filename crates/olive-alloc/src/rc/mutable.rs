use super::Rc;
use super::uninit::UniqueRcUninit;
use crate::alloc::Allocator;
use olive_core::alloc::AllocatorTryClone;
use olive_core::ptr::{self};
use olive_core::try_traits::try_clone::{TryCloneError, TryCloneToUninit};

// ---------------------------------------------------------------------------
// Mutation block (?Sized)
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Gets a mutable reference to the contained value if this `Rc` is the
    /// sole strong reference **and** no `Weak` pointers to the same
    /// allocation exist. Returns `None` otherwise.
    ///
    /// This mirrors std's `Rc::get_mut`, which requires both that there are no
    /// other strong references *and* no weak references, because a live `Weak`
    /// could upgrade back into a second strong reference at any moment.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        // SAFETY: ensured by the uniqueness check below.
        if Self::is_unique(this) {
            Some(unsafe { Self::get_mut_unchecked(this) })
        } else {
            None
        }
    }

    /// Gets a mutable reference to the contained value **without** checking
    /// the strong count.
    ///
    /// # Safety
    ///
    /// If any other Rc or Weak pointers to the same allocation exist,
    /// then they must not be dereferenced or have active borrows for the
    /// duration of the returned borrow, and their inner type must be exactly
    /// the same as the inner type of this Rc (including lifetimes).
    ///
    /// This is trivially the case if no such pointers exist, for example
    /// immediately after Rc::new.
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        let inner = unsafe { Self::inner_mut(this) };
        &mut inner.value
    }

    /// Makes a mutable reference into the given `Rc`, disassociating other
    /// references by moving or cloning as needed.
    ///
    /// Three cases, mirroring std's `Rc::make_mut`:
    ///
    /// * **Unique** — no other [`Rc`] or [`Weak`] pointers exist: the payload is
    ///   accessed in place with no allocation.
    /// * **Shared** — more than one strong reference exists: the inner value is
    ///   cloned into a fresh allocation via [`TryCloneToUninit`], and this [`Rc`]
    ///   is replaced in place to point at the clone. The old shared allocation keeps
    ///   serving the remaining owners.
    /// * **Only weak refs remain** — exactly one strong reference (this one) but
    ///   some [`Weak`] pointers: the value is moved out of the old block into a
    ///   freshly allocated one, and the old block's counts are decremented, so its
    ///   payload is logically gone. The surviving `Weak`s no longer points to a live
    ///   value (their `upgrade` will now fail). No clone is performed.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning the allocator handle, allocating the
    /// replacement block, or the clone itself fails.
    #[inline]
    pub fn try_make_mut(this: &mut Self) -> Result<&mut T, TryCloneError>
    where
        T: TryCloneToUninit,
        A: AllocatorTryClone,
    {
        // Case 1: shared (more than one strong reference) — clone the payload
        // into a fresh block and reassign `this` to the clone.
        if Self::strong_count(this) != 1 {
            let alloc = A::try_clone(&this.alloc)?;
            let new_rc = Rc::try_clone_from_ref_in(&**this, alloc)?;
            *this = new_rc;
            // SAFETY: `new_rc` was just constructed with strong == 1 and weak == 0 (excluding the implicit ref),
            // so it is uniquely owned.
            return Ok(unsafe { Self::get_mut_unchecked(this) });
        }

        // Case 2: unique — mutate in place. We just ensured strong == 1 above.
        if Self::weak_count(this) == 0 {
            // SAFETY: strong == 1 and weak == 0 (excluding the implicit ref),
            // so we are the exclusive owner.
            return Ok(unsafe { Self::get_mut_unchecked(this) });
        }

        // Case 3: strong == 1 but weak > 0 — steal the data into a fresh block.
        let size_of_val = size_of_val::<T>(&**this);
        let alloc = A::try_clone(&this.alloc)?;
        let in_progress = UniqueRcUninit::try_new_for_value(&**this, alloc)?;
        // SAFETY: we own the only strong reference, so moving the payload out by
        // byte-copying `size_of_val` bytes is sound for any `T: ?Sized`. After
        // this copy the source slot is treated as moved-from; the counters are
        // adjusted immediately below so the old block never drops the payload.
        unsafe {
            // Almost all the block below all the way until drop() will not panic.
            ptr::copy_nonoverlapping(
                Self::as_ptr(this).cast::<u8>(),
                in_progress.data_ptr().cast::<u8>(),
                size_of_val,
            );

            // Leave the old block with 0 strong refs: the data has effectively
            // been moved to the new rc. Unreachable underflow cannot occur — we
            // hold a live strong reference, so the count is at least one.
            Self::inner(this)
                .dec_strong()
                .expect("strong count underflow");

            // Remove the implicit strong-held weak ref. Other `Weak`s remain and
            // are responsible for freeing the (now empty) block.
            Self::inner(this).dec_weak().expect("weak count underflow");

            // Last chance to not accidentally forget the allocator before we
            // overwrite `this`. The `_alloc` will be dropped last after the state has
            // committed to avoid panics.
            let alloc = ptr::read(&this.alloc);

            // Replace `this` with the freshly constructed `Rc` holding the moved
            // data. Writing over `this` using pointer syntax inhibits its `Drop` impl,
            // so the old Rc is not dropped.
            // SAFETY: we already transferred the data.
            ptr::write(this, in_progress.into_rc());

            // Panics may occur here, but the state is already fully committed at this point.
            drop(alloc);
        }

        // SAFETY: after the move-out, `this.ptr` is the *only* pointer to the
        // new allocation (strong == 1, weak == 0), and we required the `Rc<T>`
        // itself to be `mut`, so this is the only possible reference to the
        // payload.
        Ok(unsafe { Self::get_mut_unchecked(this) })
    }
}
