//! Runtime: engine glue that ties tokenizer + graph + backend + KV cache
//! into a single-request generation loop.
//!
//! Phase 1 keeps this deliberately simple:
//!   * one request at a time
//!   * contiguous KV cache (paged allocator lands in Phase 3)
//!   * synchronous loop (no tokio)
//!   * greedy or top-k/top-p sampling
//!
//! The public entry point is [`Engine`].

#![deny(missing_docs)]

pub mod engine;
pub mod executor;
pub mod kv_cache;

pub use engine::{Engine, EngineBuilder, GenParams, TokenStream};
pub use kv_cache::KvCache;
