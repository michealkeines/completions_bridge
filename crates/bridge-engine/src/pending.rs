//! Pending checkpoints (ARCHITECTURE.md section 17.1, gap G2).
//!
//! A turn that will be frozen publishes a *promise* of its checkpoint before its `Done`
//! reaches the caller. A request arriving while that checkpoint is still being frozen
//! can then wait (bounded) for it instead of planning around it.
//!
//! The registry is in memory only. Every promise is resolved exactly once: explicitly
//! with the indexed record, or as `Failed` when its [`PendingGuard`] is dropped
//! (freeze error, discard, cancellation, panic, early return).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bridge_types::CheckpointKind;
use tokio::sync::watch;

use crate::lock;
use crate::store::CheckpointRecord;

/// State of one promise as seen by waiters.
#[derive(Debug, Clone)]
pub(crate) enum PendingState {
    InFlight,
    Ready(Box<CheckpointRecord>),
    Failed,
}

struct Entry {
    backend: String,
    key: String,
    kind: CheckpointKind,
    pending_call_ids: Vec<String>,
    rx: watch::Receiver<PendingState>,
}

/// A live promise matching a request prefix.
pub(crate) struct PendingMatch {
    /// Number of request messages the promised checkpoint covers.
    pub depth: usize,
    pub kind: CheckpointKind,
    pub pending_call_ids: Vec<String>,
    pub rx: watch::Receiver<PendingState>,
}

impl PendingMatch {
    /// Wait until the promise is resolved. `None` = failed (or the promiser vanished).
    /// Not bounded: callers wrap it in their own timeout / cancellation.
    pub async fn wait(mut self) -> Option<CheckpointRecord> {
        let state = match self.rx.wait_for(|s| !matches!(s, PendingState::InFlight)).await {
            Ok(s) => s.clone(),
            Err(_) => PendingState::Failed,
        };
        match state {
            PendingState::Ready(rec) => Some(*rec),
            PendingState::InFlight | PendingState::Failed => None,
        }
    }
}

type Entries = Arc<Mutex<HashMap<u64, Entry>>>;

#[derive(Default)]
pub(crate) struct PendingRegistry {
    next_id: AtomicU64,
    entries: Entries,
}

impl PendingRegistry {
    /// Promise a checkpoint of `backend` with prefix hash `key`. Must be called before the
    /// caller learns that the turn is complete. Duplicate keys are allowed.
    pub fn register(
        &self,
        backend: &str,
        key: &str,
        kind: CheckpointKind,
        pending_call_ids: Vec<String>,
    ) -> PendingGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = watch::channel(PendingState::InFlight);
        lock(&self.entries).insert(
            id,
            Entry { backend: backend.to_string(), key: key.to_string(), kind, pending_call_ids, rx },
        );
        PendingGuard { entries: self.entries.clone(), id, tx: Some(tx) }
    }

    /// The deepest live promise of `backend` whose key is one of `hashes`
    /// (`hashes[k-1]` = prefix of depth `k`) and that `accept(depth, kind, ids)` allows.
    pub fn find_deepest(
        &self,
        backend: &str,
        hashes: &[String],
        accept: impl Fn(usize, CheckpointKind, &[String]) -> bool,
    ) -> Option<PendingMatch> {
        if hashes.is_empty() {
            return None;
        }
        let depth_of: HashMap<&str, usize> =
            hashes.iter().enumerate().map(|(i, h)| (h.as_str(), i + 1)).collect();
        let entries = lock(&self.entries);
        entries
            .values()
            .filter(|e| e.backend == backend)
            .filter_map(|e| depth_of.get(e.key.as_str()).map(|&d| (d, e)))
            .filter(|(d, e)| accept(*d, e.kind, &e.pending_call_ids))
            .max_by_key(|(d, _)| *d)
            .map(|(depth, e)| PendingMatch {
                depth,
                kind: e.kind,
                pending_call_ids: e.pending_call_ids.clone(),
                rx: e.rx.clone(),
            })
    }

    /// Number of unresolved promises (diagnostics, tests).
    pub fn len(&self) -> usize {
        lock(&self.entries).len()
    }
}

/// Owner side of one promise. Dropping it unresolved resolves it as `Failed`.
pub(crate) struct PendingGuard {
    entries: Entries,
    id: u64,
    tx: Option<watch::Sender<PendingState>>,
}

impl PendingGuard {
    /// Resolve with the indexed record (`Some`) or as failed (`None`).
    pub fn resolve(mut self, record: Option<CheckpointRecord>) {
        self.finish(match record {
            Some(r) => PendingState::Ready(Box::new(r)),
            None => PendingState::Failed,
        });
    }

    fn finish(&mut self, state: PendingState) {
        let Some(tx) = self.tx.take() else { return };
        // Remove first: a later lookup then finds the record in the store instead.
        lock(&self.entries).remove(&self.id);
        tx.send_replace(state);
    }
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.finish(PendingState::Failed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge_types::SnapshotInfo;
    use std::time::Duration;

    fn rec(hash: &str) -> CheckpointRecord {
        CheckpointRecord {
            id: 7,
            backend: "b".into(),
            prefix_hash: hash.into(),
            snapshot: SnapshotInfo {
                path: "/p".into(),
                kind: CheckpointKind::Idle,
                pending_call_ids: vec![],
                size_bytes: 1,
                backend_version: "v".into(),
            },
            created_at: 0,
            last_used_at: 0,
        }
    }

    fn hashes(n: usize) -> Vec<String> {
        (1..=n).map(|i| format!("h{i}")).collect()
    }

    #[tokio::test]
    async fn resolve_wakes_waiter_and_removes_entry() {
        let reg = PendingRegistry::default();
        let g = reg.register("b", "h3", CheckpointKind::Idle, vec![]);
        let m = reg.find_deepest("b", &hashes(4), |_, _, _| true).unwrap();
        assert_eq!(m.depth, 3);
        let waiter = tokio::spawn(m.wait());
        tokio::time::sleep(Duration::from_millis(20)).await;
        g.resolve(Some(rec("h3")));
        let got = waiter.await.unwrap().unwrap();
        assert_eq!(got.prefix_hash, "h3");
        assert_eq!(reg.len(), 0);
        assert!(reg.find_deepest("b", &hashes(4), |_, _, _| true).is_none());
    }

    #[tokio::test]
    async fn dropped_guard_resolves_failed() {
        let reg = PendingRegistry::default();
        let g = reg.register("b", "h2", CheckpointKind::Awaiting, vec!["c".into()]);
        let m = reg.find_deepest("b", &hashes(2), |_, _, _| true).unwrap();
        assert_eq!(m.pending_call_ids, vec!["c".to_string()]);
        drop(g);
        assert!(m.wait().await.is_none());
        assert_eq!(reg.len(), 0);
        // A waiter that subscribes after the drop... cannot: the entry is gone.
        assert!(reg.find_deepest("b", &hashes(2), |_, _, _| true).is_none());
    }

    #[tokio::test]
    async fn find_deepest_filters_backend_prefix_and_kind() {
        let reg = PendingRegistry::default();
        let _a = reg.register("b", "h1", CheckpointKind::Idle, vec![]);
        let _b = reg.register("b", "h3", CheckpointKind::Awaiting, vec!["x".into()]);
        let _c = reg.register("other", "h4", CheckpointKind::Idle, vec![]);
        let _d = reg.register("b", "zz", CheckpointKind::Idle, vec![]);
        // Deepest on "b" within the prefix is depth 3.
        assert_eq!(reg.find_deepest("b", &hashes(5), |_, _, _| true).map(|m| m.depth), Some(3));
        // Kind filter skips it.
        let idle_only = |_: usize, k: CheckpointKind, _: &[String]| k == CheckpointKind::Idle;
        assert_eq!(reg.find_deepest("b", &hashes(5), idle_only).map(|m| m.depth), Some(1));
        // A shorter request does not see the deeper promise.
        assert_eq!(reg.find_deepest("b", &hashes(2), |_, _, _| true).map(|m| m.depth), Some(1));
        assert!(reg.find_deepest("none", &hashes(5), |_, _, _| true).is_none());
        assert_eq!(reg.len(), 4);
    }

    #[tokio::test]
    async fn duplicate_keys_are_independent() {
        let reg = PendingRegistry::default();
        let g1 = reg.register("b", "h2", CheckpointKind::Idle, vec![]);
        let g2 = reg.register("b", "h2", CheckpointKind::Idle, vec![]);
        assert_eq!(reg.len(), 2);
        g1.resolve(None);
        let m = reg.find_deepest("b", &hashes(3), |_, _, _| true).unwrap();
        g2.resolve(Some(rec("h2")));
        assert!(m.wait().await.is_some());
        assert_eq!(reg.len(), 0);
    }
}
