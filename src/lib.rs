//! A lock-free concurrent slab. Supports concurrent insertion, but not removal.
//! - **This is a fork of the [sharded-slab](https://github.com/hawkw/sharded-slab) crate.**
//! - **This crate is strictly less powerful then the original one.**
//!
//! ## Differences with `shareded-slab`
//! - `remove` and `take` methods require `&mut` access to the slab.
//! - `get` method returns `&T` instead of entry guard. It is also more efficient.
//! - There is `get_mut` method that returns `&mut T`
//! - No `get_owned` method.
//! - No `Pool` type.
//!
//! ## Why use this crate over `shareded-slab`?
//!
//! If you need `get` method to be as fast as possible and/or you need `get_mut` method, and you don't need shared removal of entries.
//!
//!
//! ## Description
//! Slabs provide pre-allocated storage for many instances of a single data
//! type. When a large number of values of a single type are required,
//! this can be more efficient than allocating each item individually. Since the
//! allocated items are the same size, memory fragmentation is reduced, and
//! creating and removing new items can be very cheap.
//!
//! This crate implements a lock-free concurrent slab, indexed by `usize`s.
//!
//! ## Usage
//!
//! First, add this to your `Cargo.toml`:
//!
//! ```toml
//! sharded_slab_lite = "0.1"
//! ```
//!
//!
//! [`Slab`] implements a slab for _storing_ small types, sharing them between
//! threads, and accessing them by index. New entries are allocated by
//! [inserting] data, moving it in by value. Similarly, entries may be
//! deallocated by [taking] from the slab, moving the value out. This API is
//! similar to a `Vec<Option<T>>`, but allowing lock-free concurrent insertion.
//!
//!
//! [inserting]: Slab::insert
//! [taking]: Slab::take
//! [create]: Pool::create
//! [cleared]: Clear
//!
//! # Examples
//!
//! Inserting an item into the slab, returning an index:
//! ```rust
//! # use sharded_slab_lite::Slab;
//! let slab = Slab::new();
//!
//! let key = slab.insert("hello world").unwrap();
//! assert_eq!(*slab.get(key).unwrap(), "hello world");
//! ```
//!
//! To share a slab across threads, it may be wrapped in an `Arc`:
//! ```rust
//! # use sharded_slab_lite::Slab;
//! use std::sync::Arc;
//! let slab = Arc::new(Slab::new());
//!
//! let slab2 = slab.clone();
//! let thread2 = std::thread::spawn(move || {
//!     let key = slab2.insert("hello from thread two").unwrap();
//!     assert_eq!(*slab2.get(key).unwrap(), "hello from thread two");
//!     key
//! });
//!
//! let key1 = slab.insert("hello from thread one").unwrap();
//! assert_eq!(*slab.get(key1).unwrap(), "hello from thread one");
//!
//! // Wait for thread 2 to complete.
//! let key2 = thread2.join().unwrap();
//!
//! // The item inserted by thread 2 remains in the slab.
//! assert_eq!(*slab.get(key2).unwrap(), "hello from thread two");
//!```
//!
//!
//! # Configuration
//!
//! For performance reasons, several values used by the slab are calculated as
//! constants. In order to allow users to tune the slab's parameters, we provide
//! a [`Config`] trait which defines these parameters as associated `consts`.
//! The `Slab` type is generic over a `C: Config` parameter.
//!
//! [`Config`]: trait.Config.html
//!
//!
//! # Safety and Correctness
//!
//! Most implementations of lock-free data structures in Rust require some
//! amount of unsafe code, and this crate is not an exception. In order to catch
//! potential bugs in this unsafe code, we make use of [`loom`], a
//! permutation-testing tool for concurrent Rust programs. All `unsafe` blocks
//! this crate occur in accesses to `loom` `UnsafeCell`s. This means that when
//! those accesses occur in this crate's tests, `loom` will assert that they are
//! valid under the C11 memory model across multiple permutations of concurrent
//! executions of those tests.
//!
//! In order to guard against the [ABA problem][aba], this crate makes use of
//! _generational indices_. Each slot in the slab tracks a generation counter
//! which is incremented every time a value is inserted into that slot, and the
//! indices returned by [`Slab::insert`] include the generation of the slot when
//! the value was inserted, packed into the high-order bits of the index. This
//! ensures that if a value is inserted, removed,  and a new value is inserted
//! into the same slot in the slab, the key returned by the first call to
//! `insert` will not map to the new value.
//!
//! Since a fixed number of bits are set aside to use for storing the generation
//! counter, the counter will wrap  around after being incremented a number of
//! times. To avoid situations where a returned index lives long enough to see the
//! generation counter wrap around to the same value, it is good to be fairly
//! generous when configuring the allocation of index bits.
//!
//! [`loom`]: https://crates.io/crates/loom
//! [aba]: https://en.wikipedia.org/wiki/ABA_problem
//! [`Slab::insert`]: struct.Slab.html#method.insert
//!
//!
#![doc(html_root_url = "https://docs.rs/sharded-slab/0.1.4")]
#![warn(missing_debug_implementations, missing_docs)]
#![cfg_attr(docsrs, warn(rustdoc::broken_intra_doc_links))]
#[macro_use]
mod macros;

pub mod implementation;

pub(crate) mod cfg;
pub(crate) mod sync;

mod clear;
mod iter;
mod page;
mod shard;
mod tid;

pub use self::{
    cfg::{Config, DefaultConfig},
    clear::Clear,
    iter::UniqueIter,
};

pub(crate) use tid::Tid;

use cfg::CfgPrivate;
use std::{fmt, marker::PhantomData};

/// A sharded slab.
///
/// See the [crate-level documentation](crate) for details on using this type.
pub struct Slab<T, C: cfg::Config = DefaultConfig> {
    shards: shard::Array<Option<T>, C>,
    _cfg: PhantomData<C>,
}

/// A handle to a vacant entry in a [`Slab`].
///
/// `VacantEntry` allows constructing values with the key that they will be
/// assigned to.
///
/// # Examples
///
/// ```
/// # use sharded_slab_lite::Slab;
/// let mut slab = Slab::new();
///
/// let hello = {
///     let entry = slab.vacant_entry().unwrap();
///     let key = entry.key();
///
///     entry.insert((key, "hello"));
///     key
/// };
///
/// assert_eq!(hello, slab.get(hello).unwrap().0);
/// assert_eq!("hello", slab.get(hello).unwrap().1);
/// ```
#[derive(Debug)]
pub struct VacantEntry<'a, T, C: cfg::Config = DefaultConfig> {
    inner: page::slot::InitGuard<Option<T>, C>,
    key: usize,
    _lt: PhantomData<&'a ()>,
}

impl<T> Slab<T> {
    /// Returns a new slab with the default configuration parameters.
    pub fn new() -> Self {
        Self::new_with_config()
    }

    /// Returns a new slab with the provided configuration parameters.
    pub fn new_with_config<C: cfg::Config>() -> Slab<T, C> {
        C::validate();
        Slab {
            shards: shard::Array::new(),
            _cfg: PhantomData,
        }
    }
}

impl<T, C: cfg::Config> Slab<T, C> {
    /// The number of bits in each index which are used by the slab.
    ///
    /// If other data is packed into the `usize` indices returned by
    /// [`Slab::insert`], user code is free to use any bits higher than the
    /// `USED_BITS`-th bit freely.
    ///
    /// This is determined by the [`Config`] type that configures the slab's
    /// parameters. By default, all bits are used; this can be changed by
    /// overriding the [`Config::RESERVED_BITS`][res] constant.
    ///
    /// [res]: crate::Config#RESERVED_BITS
    pub const USED_BITS: usize = C::USED_BITS;

    /// Inserts a value into the slab, returning the integer index at which that
    /// value was inserted. This index can then be used to access the entry.
    ///
    /// If this function returns `None`, then the shard for the current thread
    /// is full and no items can be added until some are removed, or the maximum
    /// number of shards has been reached.
    ///
    /// # Examples
    /// ```rust
    /// # use sharded_slab_lite::Slab;
    /// let slab = Slab::new();
    ///
    /// let key = slab.insert("hello world").unwrap();
    /// assert_eq!(*slab.get(key).unwrap(), "hello world");
    /// ```
    pub fn insert(&self, value: T) -> Option<usize> {
        let (tid, shard) = self.shards.current();
        test_println!("insert {:?}", tid);
        let mut value = Some(value);
        shard
            .init_with(|idx, slot| {
                let generation = slot.insert(&mut value)?;
                Some(generation.pack(idx))
            })
            .map(|idx| tid.pack(idx))
    }

    /// Return a handle to a vacant entry allowing for further manipulation.
    ///
    /// This function is useful when creating values that must contain their
    /// slab index. The returned [`VacantEntry`] reserves a slot in the slab and
    /// is able to return the index of the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sharded_slab_lite::Slab;
    /// let mut slab = Slab::new();
    ///
    /// let hello = {
    ///     let entry = slab.vacant_entry().unwrap();
    ///     let key = entry.key();
    ///
    ///     entry.insert((key, "hello"));
    ///     key
    /// };
    ///
    /// assert_eq!(hello, slab.get(hello).unwrap().0);
    /// assert_eq!("hello", slab.get(hello).unwrap().1);
    /// ```
    pub fn vacant_entry(&self) -> Option<VacantEntry<'_, T, C>> {
        let (tid, shard) = self.shards.current();
        test_println!("vacant_entry {:?}", tid);
        shard.init_with(|idx, slot| {
            let inner = slot.init()?;
            let key = inner.generation().pack(tid.pack(idx));
            Some(VacantEntry {
                inner,
                key,
                _lt: PhantomData,
            })
        })
    }

    /// Remove the value at the given index in the slab, returning `true` if a
    /// value was removed.
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut slab = sharded_slab_lite::Slab::new();
    /// let key = slab.insert("hello world").unwrap();
    ///
    /// // Remove the item from the slab.
    /// assert!(slab.remove(key));
    ///
    /// // Now, the slot is empty.
    /// assert!(!slab.contains(key));
    /// ```
    ///
    /// [`take`]: Slab::take
    pub fn remove(&mut self, idx: usize) -> bool {
        // The `Drop` impl for `Entry` calls `remove_local` or `remove_remote` based
        // on where the guard was dropped from. If the dropped guard was the last one, this will
        // call `Slot::remove_value` which actually clears storage.
        let tid = C::unpack_tid(idx);

        test_println!("rm_deferred {:?}", tid);
        let shard = self.shards.get_mut(tid.as_usize());
        if tid.is_current() {
            shard.map(|shard| shard.remove_local(idx)).unwrap_or(false)
        } else {
            shard.map(|shard| shard.remove_remote(idx)).unwrap_or(false)
        }
    }

    /// Removes the value associated with the given key from the slab, returning
    /// it.
    ///
    /// If the slab does not contain a value for that key, `None` is returned
    /// instead.
    ///
    /// [`remove`]: Slab::remove
    pub fn take(&mut self, idx: usize) -> Option<T> {
        let tid = C::unpack_tid(idx);

        test_println!("rm {:?}", tid);
        let shard = self.shards.get_mut(tid.as_usize())?;
        if tid.is_current() {
            shard.take_local(idx)
        } else {
            shard.take_remote(idx)
        }
    }

    /// Return a reference to the value associated with the given key.
    ///
    /// If the slab does not contain a value for the given key, or if the
    /// maximum number of concurrent references to the slot has been reached,
    /// `None` is returned instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let slab = sharded_slab_lite::Slab::new();
    /// let key = slab.insert("hello world").unwrap();
    ///
    /// assert_eq!(*slab.get(key).unwrap(), "hello world");
    /// assert!(slab.get(12345).is_none());
    /// ```
    pub fn get(&self, key: usize) -> Option<&T> {
        let tid = C::unpack_tid(key);

        test_println!("get {:?}; current={:?}", tid, Tid::<C>::current());

        let shard = self.shards.get(tid.as_usize())?;
        shard.with_slot(key, |slot| slot.value().as_ref())
    }

    /// Return a unique reference to the value associated with the given key.
    ///
    /// If the slab does not contain a value for the given key, or if the
    /// maximum number of concurrent references to the slot has been reached,
    /// `None` is returned instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut slab = sharded_slab_lite::Slab::new();
    /// let key = slab.insert("hello world").unwrap();
    ///
    /// assert_eq!(*slab.get_mut(key).unwrap(), "hello world");
    /// assert!(slab.get_mut(12345).is_none());
    /// ```
    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        let tid = C::unpack_tid(key);

        test_println!("get {:?}; current={:?}", tid, Tid::<C>::current());

        let shard = self.shards.get_mut(tid.as_usize())?;
        shard.with_slot_mut(key, |slot| slot.value_mut().as_mut())
    }

    /// Returns `true` if the slab contains a value for the given key.
    ///
    /// # Examples
    ///
    /// ```
    /// let slab = sharded_slab_lite::Slab::new();
    ///
    /// let key = slab.insert("hello world").unwrap();
    /// assert!(slab.contains(key));
    ///
    /// let mut slab = slab;
    ///
    /// slab.take(key).unwrap();
    /// assert!(!slab.contains(key));
    /// ```
    pub fn contains(&self, key: usize) -> bool {
        self.get(key).is_some()
    }

    /// Returns an iterator over all the items in the slab.
    ///
    /// Because this iterator exclusively borrows the slab (i.e. it holds an
    /// `&mut Slab<T>`), elements will not be added or removed while the
    /// iteration is in progress.
    pub fn unique_iter(&mut self) -> iter::UniqueIter<'_, T, C> {
        let mut shards = self.shards.iter_mut();

        let (pages, slots) = match shards.next() {
            Some(shard) => {
                let mut pages = shard.iter();
                let slots = pages.next().and_then(page::Shared::iter);
                (pages, slots)
            }
            None => ([].iter(), None),
        };

        iter::UniqueIter {
            shards,
            pages,
            slots,
        }
    }
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug, C: cfg::Config> fmt::Debug for Slab<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slab")
            .field("shards", &self.shards)
            .field("config", &C::debug())
            .finish()
    }
}

unsafe impl<T: Send, C: cfg::Config> Send for Slab<T, C> {}
unsafe impl<T: Sync, C: cfg::Config> Sync for Slab<T, C> {}

// === impl VacantEntry ===

impl<T, C: cfg::Config> VacantEntry<'_, T, C> {
    /// Insert a value in the entry.
    ///
    /// To get the integer index at which this value will be inserted, use
    /// [`key`] prior to calling `insert`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sharded_slab_lite::Slab;
    /// let mut slab = Slab::new();
    ///
    /// let hello = {
    ///     let entry = slab.vacant_entry().unwrap();
    ///     let key = entry.key();
    ///
    ///     entry.insert((key, "hello"));
    ///     key
    /// };
    ///
    /// assert_eq!(hello, slab.get(hello).unwrap().0);
    /// assert_eq!("hello", slab.get(hello).unwrap().1);
    /// ```
    ///
    /// [`key`]: VacantEntry::key
    pub fn insert(mut self, val: T) {
        let value = unsafe {
            // Safety: this `VacantEntry` only lives as long as the `Slab` it was
            // borrowed from, so it cannot outlive the entry's slot.
            self.inner.value_mut()
        };
        debug_assert!(
            value.is_none(),
            "tried to insert to a slot that already had a value!"
        );
        *value = Some(val);
        let _released = unsafe {
            // Safety: again, this `VacantEntry` only lives as long as the
            // `Slab` it was borrowed from, so it cannot outlive the entry's
            // slot.
            self.inner.release()
        };
        debug_assert!(
            !_released,
            "removing a value before it was inserted should be a no-op"
        )
    }

    /// Return the integer index at which this entry will be inserted.
    ///
    /// A value stored in this entry will be associated with this key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sharded_slab_lite::*;
    /// let mut slab = Slab::new();
    ///
    /// let hello = {
    ///     let entry = slab.vacant_entry().unwrap();
    ///     let key = entry.key();
    ///
    ///     entry.insert((key, "hello"));
    ///     key
    /// };
    ///
    /// assert_eq!(hello, slab.get(hello).unwrap().0);
    /// assert_eq!("hello", slab.get(hello).unwrap().1);
    /// ```
    pub fn key(&self) -> usize {
        self.key
    }
}

// === pack ===

pub(crate) trait Pack<C: cfg::Config>: Sized {
    // ====== provided by each implementation =================================

    /// The number of bits occupied by this type when packed into a usize.
    ///
    /// This must be provided to determine the number of bits into which to pack
    /// the type.
    const LEN: usize;
    /// The type packed on the less significant side of this type.
    ///
    /// If this type is packed into the least significant bit of a usize, this
    /// should be `()`, which occupies no bytes.
    ///
    /// This is used to calculate the shift amount for packing this value.
    type Prev: Pack<C>;

    // ====== calculated automatically ========================================

    /// A number consisting of `Self::LEN` 1 bits, starting at the least
    /// significant bit.
    ///
    /// This is the higest value this type can represent. This number is shifted
    /// left by `Self::SHIFT` bits to calculate this type's `MASK`.
    ///
    /// This is computed automatically based on `Self::LEN`.
    const BITS: usize = {
        let shift = 1 << (Self::LEN - 1);
        shift | (shift - 1)
    };
    /// The number of bits to shift a number to pack it into a usize with other
    /// values.
    ///
    /// This is caculated automatically based on the `LEN` and `SHIFT` constants
    /// of the previous value.
    const SHIFT: usize = Self::Prev::SHIFT + Self::Prev::LEN;

    /// The mask to extract only this type from a packed `usize`.
    ///
    /// This is calculated by shifting `Self::BITS` left by `Self::SHIFT`.
    const MASK: usize = Self::BITS << Self::SHIFT;

    fn as_usize(&self) -> usize;
    fn from_usize(val: usize) -> Self;

    #[inline(always)]
    fn pack(&self, to: usize) -> usize {
        let value = self.as_usize();
        debug_assert!(value <= Self::BITS);

        (to & !Self::MASK) | (value << Self::SHIFT)
    }

    #[inline(always)]
    fn from_packed(from: usize) -> Self {
        let value = (from & Self::MASK) >> Self::SHIFT;
        debug_assert!(value <= Self::BITS);
        Self::from_usize(value)
    }
}

impl<C: cfg::Config> Pack<C> for () {
    const BITS: usize = 0;
    const LEN: usize = 0;
    const SHIFT: usize = 0;
    const MASK: usize = 0;

    type Prev = ();

    fn as_usize(&self) -> usize {
        unreachable!()
    }
    fn from_usize(_val: usize) -> Self {
        unreachable!()
    }

    fn pack(&self, _to: usize) -> usize {
        unreachable!()
    }

    fn from_packed(_from: usize) -> Self {
        unreachable!()
    }
}

#[cfg(test)]
pub(crate) use self::tests::util as test_util;

#[cfg(test)]
mod tests;
