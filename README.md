# shareded_slab_lite

- **This is a fork of the [sharded-slab](https://github.com/hawkw/sharded-slab) crate.**
- **This crate is strictly less powerful then the original one.**

## Differences with `shareded-slab`
- `remove` and `take` methods require `&mut` access to the slab.
- `get` method returns `&T` instead of entry guard. It is also more efficient.
- There is `get_mut` method that returns `&mut T`
- No `get_owned` method.
- No `Pool` type.

## Why use this crate over `shareded-slab`?

If you need `get` method to be as fast as possible and/or you need `get_mut` method, and you don't need shared removal of entries.

## Usage
```toml
sharded_slab_lite = "0.1"
```

Check out the readme of [sharded-slab](https://github.com/hawkw/sharded-slab)

## Notes
- `sharded-slab` is experimental, this crate even more so.
