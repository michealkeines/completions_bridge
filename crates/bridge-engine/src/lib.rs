//! Engine (ARCHITECTURE.md sections 3, 6.2, 7, 9, 10, 12).
//!
//! PUBLIC API IS FROZEN: `Engine::new`, `impl ChatService for Engine`, `fake::FakeBackend::new`.
//!
//! Modules:
//! - [`norm`]: message normalisation and rolling prefix hashes (6.2).
//! - [`store`]: SQLite index of checkpoints, quota, backend state, request log (9).
//! - [`planner`]: deepest usable checkpoint, append/trigger split (3.3, 7.1).
//! - [`quota`]: quota state machine and candidate ordering (10.1, 7 step 4).
//! - [`orchestrator`]: the single request path, failover, policy, freeze (7).
//! - `pending`: promises of checkpoints still being frozen (17.1).
//! - [`fake`]: scripted backend for tests and the conformance suite.

pub mod fake;
pub mod norm;
mod orchestrator;
mod pending;
pub mod planner;
pub mod quota;
pub mod store;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bridge_types::{
    Backend, BackendConfig, BackendMode, BridgeError, CancellationToken, ChatRequest, ChatService, ChoiceEvent,
    Config, ErrorClass, LimitInfo, ModelInfo,
};
use futures::stream::BoxStream;
use tokio::sync::Semaphore;

use crate::quota::{Backoff, QuotaState, QuotaStatus};
use crate::store::{QuotaRow, Store};


pub(crate) fn now_s() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One registered backend.
pub(crate) struct Slot {
    pub backend: Arc<dyn Backend>,
    pub cfg: BackendConfig,
    pub sem: Arc<Semaphore>,
    pub in_flight: AtomicUsize,
}

/// Shared engine state; `Engine` is a thin handle so spawned tasks can own it.
pub(crate) struct Inner {
    pub config: Config,
    pub slots: BTreeMap<String, Slot>,
    pub store: Store,
    pub quota: Mutex<BTreeMap<String, QuotaState>>,
    pub auth_cooldown: Mutex<HashMap<String, u64>>,
    /// Snapshot paths used as the base of a running turn (protected from GC).
    pub in_use: Mutex<HashMap<String, usize>>,
    pub started: Instant,
    pub last_version_check: Mutex<Instant>,
    pub known_versions: Mutex<HashMap<String, String>>,
    pub last_gc: Mutex<Option<Instant>>,
    pub self_test_running: Mutex<HashSet<String>>,
    /// Checkpoints promised to callers but not yet indexed (17.1).
    pub pending: pending::PendingRegistry,
}

pub struct Engine {
    inner: Arc<Inner>,
}

impl Engine {
    /// `state_dir`: where the engine keeps its SQLite database (already created).
    /// `backends`: constructed backends, keyed by their `name()`.
    pub async fn new(
        config: &Config,
        state_dir: &Path,
        backends: Vec<Arc<dyn Backend>>,
    ) -> anyhow::Result<Arc<Self>> {
        std::fs::create_dir_all(state_dir)?;
        let store = Store::open(&state_dir.join("bridge.db"), Duration::from_millis(config.engine.store_busy_timeout_ms))?;

        let mut slots = BTreeMap::new();
        for backend in backends {
            let name = backend.name().to_string();
            let cfg = match config.backends.get(&name) {
                Some(c) => c.clone(),
                None => {
                    tracing::warn!(backend = %name, "backend has no config entry; using defaults");
                    BackendConfig::default()
                }
            };
            let permits = cfg.max_concurrency.max(1) as usize;
            slots.insert(
                name,
                Slot { backend, cfg, sem: Arc::new(Semaphore::new(permits)), in_flight: AtomicUsize::new(0) },
            );
        }

        let mut quota = BTreeMap::new();
        for (name, row) in store.load_quota()? {
            if slots.contains_key(&name) {
                quota.insert(
                    name,
                    QuotaState {
                        status: QuotaStatus::parse(&row.status),
                        utilization: row.utilization,
                        resets_at: row.resets_at,
                        backoff_s: row.backoff_s,
                    },
                );
            }
        }

        let inner = Arc::new(Inner {
            config: config.clone(),
            slots,
            store,
            quota: Mutex::new(quota),
            auth_cooldown: Mutex::new(HashMap::new()),
            in_use: Mutex::new(HashMap::new()),
            started: Instant::now(),
            last_version_check: Mutex::new(Instant::now()),
            known_versions: Mutex::new(HashMap::new()),
            last_gc: Mutex::new(None),
            self_test_running: Mutex::new(HashSet::new()),
            pending: pending::PendingRegistry::default(),
        });

        // Self-test every backend concurrently.
        let tests = inner.slots.keys().cloned().map(|name| {
            let inner = inner.clone();
            async move { inner.run_self_test(&name).await }
        });
        futures::future::join_all(tests).await;

        inner.maybe_gc(true).await;
        Ok(Arc::new(Engine { inner }))
    }

    /// Run checkpoint GC now (normally throttled to `engine.gc_interval_s`).
    pub async fn gc_now(&self) {
        self.inner.maybe_gc(true).await;
    }

    /// Number of indexed checkpoints and their total size in bytes.
    pub fn checkpoint_stats(&self) -> (u64, u64) {
        self.inner.store.checkpoint_stats().unwrap_or((0, 0))
    }

    /// Most recent request log rows, newest first.
    pub fn recent_requests(&self, limit: usize) -> Vec<store::RequestLogRow> {
        self.inner.store.recent_requests(limit).unwrap_or_default()
    }

    /// Force the next request to re-check backend versions (normally every `engine.version_check_interval_s`).
    pub fn expire_version_check(&self) {
        if let Some(past) = Instant::now().checked_sub(self.inner.version_check_interval() * 2) {
            *lock(&self.inner.last_version_check) = past;
        }
    }
}

impl Inner {
    /// Minimum time between backend version re-checks (`engine.version_check_interval_s`).
    pub(crate) fn version_check_interval(&self) -> Duration {
        Duration::from_secs(self.config.engine.version_check_interval_s)
    }

    pub(crate) async fn run_self_test(self: &Arc<Self>, name: &str) {
        let Some(slot) = self.slots.get(name) else { return };
        if !lock(&self.self_test_running).insert(name.to_string()) {
            return;
        }
        let version = slot.backend.version();
        let started = Instant::now();
        let mode = slot.backend.self_test().await;
        let mode_str = mode_str(mode);
        tracing::info!(backend = %name, %version, mode = mode_str, ms = started.elapsed().as_millis() as u64, "backend self-test");
        lock(&self.known_versions).insert(name.to_string(), version.clone());
        if let Err(e) = self.store.save_backend_state(name, mode_str, &version, now_s(), mode_str) {
            tracing::warn!(error = %e, "saving backend state failed");
        }
        lock(&self.self_test_running).remove(name);
    }

    /// Re-run self-tests (in the background) for backends whose version changed.
    pub(crate) fn maybe_recheck_versions(self: &Arc<Self>) {
        {
            let mut last = lock(&self.last_version_check);
            if last.elapsed() < self.version_check_interval() {
                return;
            }
            *last = Instant::now();
        }
        for (name, slot) in &self.slots {
            let current = slot.backend.version();
            let changed = lock(&self.known_versions).get(name).is_none_or(|v| *v != current);
            if changed {
                tracing::info!(backend = %name, version = %current, "backend version changed; re-running self-test");
                let inner = self.clone();
                let name = name.clone();
                tokio::spawn(async move { inner.run_self_test(&name).await });
            }
        }
    }

    // ------------------------------------------------------------ quota

    pub(crate) fn quota_status(&self, name: &str) -> QuotaStatus {
        let now = now_s();
        let (status, changed, row) = {
            let mut q = lock(&self.quota);
            let st = q.entry(name.to_string()).or_default();
            let changed = st.tick(now);
            (st.status, changed, quota_row(st))
        };
        if changed {
            self.persist_quota(name, &row);
        }
        status
    }

    pub(crate) fn quota_retry_after(&self, names: &[String]) -> Option<u64> {
        let now = now_s();
        let q = lock(&self.quota);
        names.iter().filter_map(|n| q.get(n).and_then(|s| s.retry_after(now))).min()
    }

    fn update_quota(&self, name: &str, f: impl FnOnce(&mut QuotaState)) {
        let row = {
            let mut q = lock(&self.quota);
            let st = q.entry(name.to_string()).or_default();
            let before = st.clone();
            f(st);
            if *st == before {
                return;
            }
            tracing::info!(backend = %name, status = st.status.as_str(), resets_at = ?st.resets_at, "quota state");
            quota_row(st)
        };
        self.persist_quota(name, &row);
    }

    fn persist_quota(&self, name: &str, row: &QuotaRow) {
        if let Err(e) = self.store.save_quota(name, row, now_s()) {
            tracing::warn!(error = %e, "saving quota failed");
        }
    }

    pub(crate) fn on_limit(&self, name: &str, info: &LimitInfo) {
        let now = now_s();
        let backoff = Backoff::from_config(&self.config.engine);
        self.update_quota(name, |q| q.on_limit(info, now, backoff));
    }

    pub(crate) fn on_success(&self, name: &str) {
        self.update_quota(name, |q| q.on_success());
    }

    /// Record what a backend failure means for routing.
    pub(crate) fn on_failure(&self, name: &str, err: &BridgeError) {
        match err.class {
            ErrorClass::QuotaExhausted => {
                let now = now_s();
                let backoff = Backoff::from_config(&self.config.engine);
                self.update_quota(name, |q| q.on_quota_error(err.retry_after, now, backoff));
            }
            ErrorClass::BackendAuth => {
                lock(&self.auth_cooldown).insert(name.to_string(), now_s().saturating_add(self.config.engine.auth_cooldown_s));
            }
            _ => {}
        }
    }

    pub(crate) fn in_auth_cooldown(&self, name: &str) -> bool {
        let now = now_s();
        let mut c = lock(&self.auth_cooldown);
        match c.get(name) {
            Some(&until) if until > now => true,
            Some(_) => {
                c.remove(name);
                false
            }
            None => false,
        }
    }

    // ------------------------------------------------------------ checkpoints

    pub(crate) fn mark_in_use(&self, path: &str) {
        *lock(&self.in_use).entry(path.to_string()).or_insert(0) += 1;
    }

    pub(crate) fn unmark_in_use(&self, path: &str) {
        let mut m = lock(&self.in_use);
        if let Some(c) = m.get_mut(path) {
            *c -= 1;
            if *c == 0 {
                m.remove(path);
            }
        }
    }

    /// LRU + TTL garbage collection; throttled unless `force`.
    pub(crate) async fn maybe_gc(&self, force: bool) {
        {
            let mut last = lock(&self.last_gc);
            if !force && last.is_some_and(|t| t.elapsed() < Duration::from_secs(self.config.engine.gc_interval_s)) {
                // The interval only throttles TTL sweeps; the size cap is enforced as soon as
                // it is exceeded (heavy use can write far more than the cap within one interval).
                let max_bytes = self.config.checkpoints.max_total_mb.saturating_mul(1024 * 1024);
                let over_cap = self.store.checkpoint_stats().map(|(_, bytes)| bytes > max_bytes).unwrap_or(false);
                if !over_cap {
                    return;
                }
            }
            *last = Some(Instant::now());
        }
        let cfg = &self.config.checkpoints;
        let ttl_ms = i64::try_from(cfg.ttl_hours.saturating_mul(3_600_000)).unwrap_or(i64::MAX);
        let max_bytes = cfg.max_total_mb.saturating_mul(1024 * 1024);
        let in_use: HashSet<String> = lock(&self.in_use).keys().cloned().collect();
        let removed = match self.store.gc(now_ms(), ttl_ms, max_bytes, &in_use) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "checkpoint gc failed");
                return;
            }
        };
        if !removed.is_empty() {
            tracing::info!(count = removed.len(), "checkpoint gc removed checkpoints");
        }
        for rec in removed {
            if let Some(slot) = self.slots.get(&rec.backend) {
                slot.backend.delete_snapshot(&rec.snapshot).await;
            }
        }
    }
}

fn quota_row(st: &QuotaState) -> QuotaRow {
    QuotaRow {
        status: st.status.as_str().to_string(),
        utilization: st.utilization,
        resets_at: st.resets_at,
        backoff_s: st.backoff_s,
    }
}

pub(crate) fn mode_str(mode: BackendMode) -> &'static str {
    match mode {
        BackendMode::Native => "native",
        BackendMode::Transcript => "transcript",
        BackendMode::Disabled => "disabled",
    }
}

#[async_trait]
impl ChatService for Engine {
    fn models(&self) -> Vec<ModelInfo> {
        let inner = &self.inner;
        inner
            .config
            .models
            .iter()
            .map(|(id, route)| {
                let context_limit = route.context_limit.or_else(|| match &route.backend {
                    Some(b) => inner.config.backends.get(b).map(|c| c.context_limit),
                    None => inner
                        .config
                        .backends
                        .values()
                        .filter(|c| c.enabled)
                        .map(|c| c.context_limit)
                        .min(),
                });
                ModelInfo { id: id.clone(), created: inner.config.engine.model_created_unix, owned_by: "completions-bridge".into(), context_limit }
            })
            .collect()
    }

    async fn chat(&self, request: ChatRequest, cancel: CancellationToken) -> BoxStream<'static, ChoiceEvent> {
        orchestrator::chat(self.inner.clone(), request, cancel)
    }

    fn status(&self) -> serde_json::Value {
        let inner = &self.inner;
        let now = now_s();
        let backends: Vec<serde_json::Value> = inner
            .slots
            .iter()
            .map(|(name, slot)| {
                let status = inner.quota_status(name);
                let q = lock(&inner.quota).get(name).cloned().unwrap_or_default();
                serde_json::json!({
                    "name": name,
                    "enabled": slot.cfg.enabled,
                    "priority": slot.cfg.priority,
                    "mode": mode_str(slot.backend.mode()),
                    "version": slot.backend.version(),
                    "quota": {
                        "status": status.as_str(),
                        "utilization": q.utilization,
                        "resets_at": q.resets_at,
                        "retry_after_s": q.retry_after(now),
                    },
                    "in_flight": slot.in_flight.load(Ordering::SeqCst),
                    "max_concurrency": slot.cfg.max_concurrency,
                })
            })
            .collect();
        let (count, size) = inner.store.checkpoint_stats().unwrap_or((0, 0));
        serde_json::json!({
            "uptime_s": inner.started.elapsed().as_secs(),
            "backends": backends,
            "checkpoints": { "count": count, "size_bytes": size, "pending": inner.pending.len() },
        })
    }
}
