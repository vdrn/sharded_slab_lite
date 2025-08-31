use super::FreeList;
use crate::sync::{
    UnsafeCell,
    atomic::{AtomicUsize, Ordering},
};
use crate::{Pack, Tid, cfg, clear::Clear};
use std::{fmt, marker::PhantomData, mem, ptr, thread};

pub(crate) struct Slot<T, C> {
    lifecycle: AtomicUsize,
    /// The offset of the next item on the free list.
    next: UnsafeCell<usize>,
    /// The data stored in the slot.
    item: UnsafeCell<T>,
    _cfg: PhantomData<fn(C)>,
}

// #[derive(Debug)]
// pub(crate) struct Guard<T, C: cfg::Config = cfg::DefaultConfig> {
//     slot: ptr::NonNull<Slot<T, C>>,
// }

#[derive(Debug)]
pub(crate) struct InitGuard<T, C: cfg::Config = cfg::DefaultConfig> {
    slot: ptr::NonNull<Slot<T, C>>,
    curr_lifecycle: usize,
    released: bool,
}

#[repr(transparent)]
pub(crate) struct Generation<C = cfg::DefaultConfig> {
    value: usize,
    _cfg: PhantomData<fn(C)>,
}

#[repr(transparent)]
pub(crate) struct RefCount<C = cfg::DefaultConfig> {
    value: usize,
    _cfg: PhantomData<fn(C)>,
}

pub(crate) struct Lifecycle<C> {
    state: State,
    _cfg: PhantomData<fn(C)>,
}
struct LifecycleGen<C>(Generation<C>);

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
#[repr(usize)]
enum State {
    Present = 0b00,
    Marked = 0b01,
    Removing = 0b11,
}

impl<C: cfg::Config> Pack<C> for Generation<C> {
    /// Use all the remaining bits in the word for the generation counter, minus
    /// any bits reserved by the user.
    const LEN: usize = (cfg::WIDTH - C::RESERVED_BITS) - Self::SHIFT;

    type Prev = Tid<C>;

    #[inline(always)]
    fn from_usize(u: usize) -> Self {
        debug_assert!(u <= Self::BITS);
        Self::new(u)
    }

    #[inline(always)]
    fn as_usize(&self) -> usize {
        self.value
    }
}

impl<C: cfg::Config> Generation<C> {
    fn new(value: usize) -> Self {
        Self {
            value,
            _cfg: PhantomData,
        }
    }
}

// Slot methods which should work across all trait bounds
impl<T, C> Slot<T, C>
where
    C: cfg::Config,
{
    #[inline(always)]
    pub(super) fn next(&self) -> usize {
        self.next.with(|next| unsafe { *next })
    }

    #[inline(always)]
    pub(crate) fn value(&self) -> &T {
        self.item.with(|item| unsafe { &*item })
    }
    #[inline(always)]
    pub(crate) fn value_mut(&mut self) -> &mut T {
        self.item.with_mut(|item| unsafe { &mut *item })
    }

    #[inline(always)]
    pub(super) fn set_next(&self, next: usize) {
        self.next.with_mut(|n| unsafe {
            (*n) = next;
        })
    }

    /// Mutates this slot.
    fn release_with<F, M, R>(&mut self, offset: usize, free: &F, mutator: M) -> R
    where
        F: FreeList<C>,
        M: FnOnce(&mut T) -> R,
    {
        let res = mutator(self.item.get_mut());
        free.push(offset, self);
        res
    }

    /// Initialize a slot
    ///
    /// This method initializes and sets up the state for a slot. When being used in `Pool`, we
    /// only need to ensure that the `Slot` is in the right `state, while when being used in a
    /// `Slab` we want to insert a value into it, as the memory is not initialized
    pub(crate) fn init(&self) -> Option<InitGuard<T, C>> {
        // Load the current lifecycle state.
        let lifecycle = self.lifecycle.load(Ordering::Acquire);
        let generation = LifecycleGen::<C>::from_packed(lifecycle).0;
        let refs = RefCount::<C>::from_packed(lifecycle);

        test_println!(
            "-> initialize_state; state={:?}; gen={:?}; refs={:?};",
            Lifecycle::<C>::from_packed(lifecycle),
            generation,
            refs,
        );

        if refs.value != 0 {
            test_println!("-> initialize while referenced! cancelling");
            return None;
        }

        Some(InitGuard {
            slot: ptr::NonNull::from(self),
            curr_lifecycle: lifecycle,
            released: false,
        })
    }
}

// Slot impl which _needs_ an `Option` for self.item, this is for `Slab` to use.
impl<T, C> Slot<Option<T>, C>
where
    C: cfg::Config,
{
    fn is_empty(&self) -> bool {
        self.item.with(|item| unsafe { (*item).is_none() })
    }

    /// Insert a value into a slot
    ///
    /// We first initialize the state and then insert the pased in value into the slot.
    #[inline]
    pub(crate) fn insert(&self, value: &mut Option<T>) -> Option<Generation<C>> {
        debug_assert!(self.is_empty(), "inserted into full slot");
        debug_assert!(value.is_some(), "inserted twice");

        let mut guard = self.init()?;
        let generation = guard.generation();
        unsafe {
            // Safety: Accessing the value of an `InitGuard` is unsafe because
            // it has a pointer to a slot which may dangle. Here, we know the
            // pointed slot is alive because we have a reference to it in scope,
            // and the `InitGuard` will be dropped when this function returns.
            mem::swap(guard.value_mut(), value);
            guard.release();
        };
        test_println!("-> inserted at {:?}", generation);

        Some(generation)
    }

    #[inline]
    pub(super) fn remove_value<F: FreeList<C>>(&mut self, offset: usize, free: &F) -> Option<T> {
        self.release_with(offset, free, |item| item.take())
    }
}

// These impls are specific to `Pool`
impl<T, C> Slot<T, C>
where
    T: Default + Clear,
    C: cfg::Config,
{
    pub(in crate::page) fn new(next: usize) -> Self {
        Self {
            lifecycle: AtomicUsize::new(Lifecycle::<C>::REMOVING.as_usize()),
            item: UnsafeCell::new(T::default()),
            next: UnsafeCell::new(next),
            _cfg: PhantomData,
        }
    }
}

impl<T, C: cfg::Config> fmt::Debug for Slot<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lifecycle = self.lifecycle.load(Ordering::Relaxed);
        f.debug_struct("Slot")
            .field("lifecycle", &format_args!("{:#x}", lifecycle))
            .field("state", &Lifecycle::<C>::from_packed(lifecycle).state)
            .field("gen", &LifecycleGen::<C>::from_packed(lifecycle).0)
            .field("refs", &RefCount::<C>::from_packed(lifecycle))
            .field("next", &self.next())
            .finish()
    }
}

// === impl Generation ===

impl<C> fmt::Debug for Generation<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Generation").field(&self.value).finish()
    }
}

impl<C: cfg::Config> PartialEq for Generation<C> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<C: cfg::Config> Eq for Generation<C> {}

impl<C: cfg::Config> PartialOrd for Generation<C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: cfg::Config> Ord for Generation<C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<C: cfg::Config> Clone for Generation<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: cfg::Config> Copy for Generation<C> {}

// === impl Lifecycle ===

impl<C: cfg::Config> Lifecycle<C> {
    const REMOVING: Self = Self {
        state: State::Removing,
        _cfg: PhantomData,
    };
    const PRESENT: Self = Self {
        state: State::Present,
        _cfg: PhantomData,
    };
}

impl<C: cfg::Config> Pack<C> for Lifecycle<C> {
    const LEN: usize = 2;
    type Prev = ();

    fn from_usize(u: usize) -> Self {
        Self {
            state: match u & Self::MASK {
                0b00 => State::Present,
                0b01 => State::Marked,
                0b11 => State::Removing,
                bad => unreachable!("weird lifecycle {:#b}", bad),
            },
            _cfg: PhantomData,
        }
    }

    fn as_usize(&self) -> usize {
        self.state as usize
    }
}

impl<C> PartialEq for Lifecycle<C> {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
    }
}

impl<C> Eq for Lifecycle<C> {}

impl<C> fmt::Debug for Lifecycle<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Lifecycle").field(&self.state).finish()
    }
}

// === impl RefCount ===

impl<C: cfg::Config> Pack<C> for RefCount<C> {
    const LEN: usize = cfg::WIDTH - (Lifecycle::<C>::LEN + Generation::<C>::LEN);
    type Prev = Lifecycle<C>;

    fn from_usize(value: usize) -> Self {
        debug_assert!(value <= Self::BITS);
        Self {
            value,
            _cfg: PhantomData,
        }
    }

    fn as_usize(&self) -> usize {
        self.value
    }
}

impl<C: cfg::Config> RefCount<C> {
    pub(crate) const MAX: usize = Self::BITS - 1;
}

impl<C> fmt::Debug for RefCount<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RefCount").field(&self.value).finish()
    }
}

impl<C: cfg::Config> PartialEq for RefCount<C> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<C: cfg::Config> Eq for RefCount<C> {}

impl<C: cfg::Config> PartialOrd for RefCount<C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: cfg::Config> Ord for RefCount<C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<C: cfg::Config> Clone for RefCount<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: cfg::Config> Copy for RefCount<C> {}

// === impl LifecycleGen ===

impl<C: cfg::Config> Pack<C> for LifecycleGen<C> {
    const LEN: usize = Generation::<C>::LEN;
    type Prev = RefCount<C>;

    fn from_usize(value: usize) -> Self {
        Self(Generation::from_usize(value))
    }

    fn as_usize(&self) -> usize {
        self.0.as_usize()
    }
}

impl<T, C: cfg::Config> InitGuard<T, C> {
    pub(crate) fn generation(&self) -> Generation<C> {
        LifecycleGen::<C>::from_packed(self.curr_lifecycle).0
    }

    /// Returns a mutably borrowed reference to the slot's value.
    ///
    /// ## Safety
    ///
    /// This dereferences a raw pointer to the slot. The caller is responsible
    /// for ensuring that the `InitGuard` does not outlive the slab that
    /// contains the pointed slot. Failure to do so means this pointer may
    /// dangle.
    ///
    /// It's safe to reference the slot mutably, though, because creating an
    /// `InitGuard` ensures there are no outstanding immutable references.
    pub(crate) unsafe fn value_mut(&mut self) -> &mut T {
        self.slot.as_ref().item.with_mut(|val| &mut *val)
    }

    /// Releases the guard, returning `true` if the slot should be cleared.
    ///
    /// ## Safety
    ///
    /// This dereferences a raw pointer to the slot. The caller is responsible
    /// for ensuring that the `InitGuard` does not outlive the slab that
    /// contains the pointed slot. Failure to do so means this pointer may
    /// dangle.
    pub(crate) unsafe fn release(&mut self) -> bool {
        self.release2(0)
    }

    unsafe fn release2(&mut self, new_refs: usize) -> bool {
        test_println!(
            "InitGuard::release; curr_lifecycle={:?}; downgrading={}",
            Lifecycle::<C>::from_packed(self.curr_lifecycle),
            new_refs != 0,
        );
        if self.released {
            test_println!("-> already released!");
            return false;
        }
        self.released = true;
        let mut curr_lifecycle = self.curr_lifecycle;
        let slot = self.slot.as_ref();
        let new_lifecycle = LifecycleGen::<C>::from_packed(self.curr_lifecycle)
            .pack(Lifecycle::<C>::PRESENT.pack(new_refs));

        match slot.lifecycle.compare_exchange(
            curr_lifecycle,
            new_lifecycle,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                test_println!("--> advanced to PRESENT; done");
                return false;
            }
            Err(actual) => {
                test_println!(
                    "--> lifecycle changed; actual={:?}",
                    Lifecycle::<C>::from_packed(actual)
                );
                curr_lifecycle = actual;
            }
        }

        // if the state was no longer the prior state, we are now responsible
        // for releasing the slot.
        loop {
            let refs = RefCount::<C>::from_packed(curr_lifecycle);
            let state = Lifecycle::<C>::from_packed(curr_lifecycle).state;

            test_println!(
                "-> InitGuard::release; lifecycle={:#x}; state={:?}; refs={:?};",
                curr_lifecycle,
                state,
                refs,
            );

            debug_assert!(
                state == State::Marked || thread::panicking(),
                "state was not MARKED; someone else has removed the slot while we have exclusive access!\nactual={:?}",
                state
            );
            debug_assert!(
                refs.value == 0 || thread::panicking(),
                "ref count was not 0; someone else has referenced the slot while we have exclusive access!\nactual={:?}",
                refs
            );

            let new_lifecycle = LifecycleGen(self.generation()).pack(State::Removing as usize);

            match slot.lifecycle.compare_exchange(
                curr_lifecycle,
                new_lifecycle,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    test_println!("-> InitGuard::RELEASE: done!");
                    return true;
                }
                Err(actual) => {
                    debug_assert!(thread::panicking(), "we should not have to retry this CAS!");
                    test_println!("-> InitGuard::release; retry, actual={:#x}", actual);
                    curr_lifecycle = actual;
                }
            }
        }
    }
}
