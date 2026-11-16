//! Shared, frozen types for the completions bridge.
//!
//! OWNERSHIP: this crate is owned by the coordinator. Implementation agents must not
//! change it; if something is missing, work around it locally and report the need.

pub mod backend;
pub mod config;
pub mod error;
pub mod ir;
pub mod service;

pub use backend::*;
pub use config::*;
pub use error::*;
pub use ir::*;
pub use service::*;

pub use tokio_util::sync::CancellationToken;
