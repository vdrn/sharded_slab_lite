use super::util::*;
use crate::Slab;
use crate::sync::alloc;
use loom::sync::{Condvar, Mutex};
use loom::thread;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[test]
fn take_local() {
    run_model("take_local", || {
        let mut s = Slab::new();
        let t1 = thread::spawn(move || {
            let idx = s.insert(1).expect("insert");
            assert_eq!(*s.get(idx).unwrap(), 1);
            assert_eq!(s.take(idx), Some(1));
            assert!(s.get(idx).is_none());
            let idx = s.insert(2).expect("insert");
            assert_eq!(*s.get(idx).unwrap(), 2);
            assert_eq!(s.take(idx), Some(2));
            assert!(s.get(idx).is_none());
        });

        let mut s = Slab::new();
        let t2 = thread::spawn(move || {
            let idx = s.insert(3).expect("insert");
            assert_eq!(*s.get(idx).unwrap(), 3);
            assert_eq!(s.take(idx), Some(3));
            assert!(s.get(idx).is_none());
            let idx = s.insert(4).expect("insert");
            assert_eq!(*s.get(idx).unwrap(), 4);
            assert_eq!(s.take(idx), Some(4));
            assert!(s.get(idx).is_none());
        });

        let mut s = Slab::new();
        let idx1 = s.insert(5).expect("insert");
        assert_eq!(*s.get(idx1).unwrap(), 5);
        let idx2 = s.insert(6).expect("insert");
        assert_eq!(*s.get(idx2).unwrap(), 6);
        assert_eq!(s.take(idx1), Some(5));
        assert!(s.get(idx1).is_none());
        assert_eq!(*s.get(idx2).unwrap(), 6);
        assert_eq!(s.take(idx2), Some(6));
        assert!(s.get(idx2).is_none());

        t1.join().expect("thread 1 should not panic");
        t2.join().expect("thread 2 should not panic");
    });
}

fn store_when_free<C: crate::Config>(slab: &Arc<Slab<usize, C>>, t: usize) -> usize {
    loop {
        test_println!("try store {:?}", t);
        if let Some(key) = slab.insert(t) {
            test_println!("inserted at {:#x}", key);
            return key;
        }
        test_println!("retrying; slab is full...");
        thread::yield_now();
    }
}

struct TinierConfig;

impl crate::Config for TinierConfig {
    const INITIAL_PAGE_SIZE: usize = 2;
    const MAX_PAGES: usize = 1;
}

struct SetDropped {
    val: usize,
    dropped: std::sync::Arc<AtomicBool>,
}

struct AssertDropped {
    dropped: std::sync::Arc<AtomicBool>,
}

impl AssertDropped {
    fn new(val: usize) -> (Self, SetDropped) {
        let dropped = std::sync::Arc::new(AtomicBool::new(false));
        let val = SetDropped {
            val,
            dropped: dropped.clone(),
        };
        (Self { dropped }, val)
    }

    fn assert_dropped(&self) {
        assert!(
            self.dropped.load(Ordering::SeqCst),
            "value should have been dropped!"
        );
    }
}

impl Drop for SetDropped {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn unique_iter() {
    run_model("unique_iter", || {
        let mut slab = Arc::new(Slab::new());

        let s = slab.clone();
        let t1 = thread::spawn(move || {
            s.insert(1).expect("insert");
            s.insert(2).expect("insert");
        });

        let s = slab.clone();
        let t2 = thread::spawn(move || {
            s.insert(3).expect("insert");
            s.insert(4).expect("insert");
        });

        t1.join().expect("thread 1 should not panic");
        t2.join().expect("thread 2 should not panic");

        let slab = Arc::get_mut(&mut slab).expect("other arcs should be dropped");
        let items: Vec<_> = slab.unique_iter().map(|&i| i).collect();
        assert!(items.contains(&1), "items: {:?}", items);
        assert!(items.contains(&2), "items: {:?}", items);
        assert!(items.contains(&3), "items: {:?}", items);
        assert!(items.contains(&4), "items: {:?}", items);
    });
}

#[test]
fn custom_page_sz() {
    let mut model = loom::model::Builder::new();
    model.max_branches = 100000;
    model.check(|| {
        let slab = Slab::<usize>::new_with_config::<TinyConfig>();

        for i in 0..1024usize {
            test_println!("{}", i);
            let k = slab.insert(i).expect("insert");
            let v = *slab.get(k).expect("get");
            assert_eq!(v, i, "slab: {:#?}", slab);
        }
    });
}

mod free_list_reuse {
    use super::*;
    struct TinyConfig;

    impl crate::cfg::Config for TinyConfig {
        const INITIAL_PAGE_SIZE: usize = 2;
    }

    #[test]
    fn local_remove() {
        run_model("free_list_reuse::local_remove", || {
            let mut slab = Slab::new_with_config::<TinyConfig>();

            let t1 = slab.insert("hello").expect("insert");
            let t2 = slab.insert("world").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t1).1,
                0,
                "1st slot should be on 0th page"
            );
            assert_eq!(
                crate::page::indices::<TinyConfig>(t2).1,
                0,
                "2nd slot should be on 0th page"
            );
            let t3 = slab.insert("earth").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t3).1,
                1,
                "3rd slot should be on 1st page"
            );

            slab.remove(t2);
            let t4 = slab.insert("universe").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t4).1,
                0,
                "2nd slot should be reused (0th page)"
            );

            slab.remove(t1);
            let _ = slab.insert("goodbye").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t4).1,
                0,
                "1st slot should be reused (0th page)"
            );
        });
    }

    #[test]
    fn local_take() {
        run_model("free_list_reuse::local_take", || {
            let mut slab = Slab::new_with_config::<TinyConfig>();

            let t1 = slab.insert("hello").expect("insert");
            let t2 = slab.insert("world").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t1).1,
                0,
                "1st slot should be on 0th page"
            );
            assert_eq!(
                crate::page::indices::<TinyConfig>(t2).1,
                0,
                "2nd slot should be on 0th page"
            );
            let t3 = slab.insert("earth").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t3).1,
                1,
                "3rd slot should be on 1st page"
            );

            assert_eq!(slab.take(t2), Some("world"));
            let t4 = slab.insert("universe").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t4).1,
                0,
                "2nd slot should be reused (0th page)"
            );

            assert_eq!(slab.take(t1), Some("hello"));
            let _ = slab.insert("goodbye").expect("insert");
            assert_eq!(
                crate::page::indices::<TinyConfig>(t4).1,
                0,
                "1st slot should be reused (0th page)"
            );
        });
    }
}

#[test]
fn vacant_entry() {
    run_model("vacant_entry", || {
        let slab = Arc::new(Slab::new());
        let entry = slab.vacant_entry().unwrap();
        let key: usize = entry.key();

        let slab2 = slab.clone();
        let t1 = thread::spawn(move || {
            test_dbg!(slab2.get(key));
        });

        entry.insert("hello world");
        t1.join().unwrap();

        assert_eq!(*slab.get(key).expect("get"), "hello world");
    });
}

#[test]
fn vacant_entry_2() {
    run_model("vacant_entry_2", || {
        let slab = Arc::new(Slab::new());
        let entry = slab.vacant_entry().unwrap();
        let key: usize = entry.key();

        let slab2 = slab.clone();
        let slab3 = slab.clone();
        let t1 = thread::spawn(move || {
            test_dbg!(slab2.get(key));
        });

        entry.insert("hello world");
        let t2 = thread::spawn(move || {
            test_dbg!(slab3.get(key));
        });

        t1.join().unwrap();
        t2.join().unwrap();
        assert_eq!(*slab.get(key).expect("get"), "hello world");
    });
}
