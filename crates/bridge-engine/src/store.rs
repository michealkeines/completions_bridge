//! SQLite persistence (ARCHITECTURE.md section 9): checkpoint index, quota state,
//! backend state and the request log.
//!
//! Concurrency choice: one `rusqlite::Connection` behind a `std::sync::Mutex`, used
//! briefly and synchronously from async code. Every statement is a tiny indexed query on
//! a local file (WAL mode), so the lock is held for microseconds to a few milliseconds;
//! this is cheaper than a `spawn_blocking` hop per query and keeps `status()` (which is
//! synchronous) simple. The mutex is never held across an `.await`.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::Context;
use bridge_types::{CheckpointKind, SnapshotInfo};
use rusqlite::{params, Connection, OptionalExtension};

/// One indexed checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointRecord {
    pub id: i64,
    pub backend: String,
    pub prefix_hash: String,
    pub snapshot: SnapshotInfo,
    /// Unix milliseconds.
    pub created_at: i64,
    pub last_used_at: i64,
}

/// Persisted quota state of one backend.
#[derive(Debug, Clone, PartialEq)]
pub struct QuotaRow {
    pub status: String,
    pub utilization: Option<f64>,
    pub resets_at: Option<u64>,
    pub backoff_s: u64,
}

/// One row of `request_log`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestLogRow {
    pub id: String,
    pub model: String,
    pub backend: Option<String>,
    pub checkpoint_hit_depth: Option<u64>,
    pub appended_messages: Option<u64>,
    pub finish_reason: Option<String>,
    pub error_class: Option<String>,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cache_read: u64,
    pub ttfb_ms: Option<u64>,
    pub total_ms: u64,
}

pub struct Store {
    conn: Mutex<Connection>,
}

fn kind_str(kind: CheckpointKind) -> &'static str {
    match kind {
        CheckpointKind::Idle => "idle",
        CheckpointKind::Awaiting => "awaiting",
    }
}

fn to_i64(v: u64) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

fn to_u64(v: i64) -> u64 {
    u64::try_from(v).unwrap_or(0)
}

impl Store {
    /// `busy_timeout`: SQLite busy timeout (`engine.store_busy_timeout_ms`).
    pub fn open(path: &Path, busy_timeout: std::time::Duration) -> anyhow::Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::init(conn, busy_timeout)
    }

    pub fn open_in_memory() -> anyhow::Result<Self> {
        let busy = std::time::Duration::from_millis(bridge_types::EngineConfig::default().store_busy_timeout_ms);
        Self::init(Connection::open_in_memory()?, busy)
    }

    fn init(conn: Connection, busy_timeout: std::time::Duration) -> anyhow::Result<Self> {
        conn.busy_timeout(busy_timeout)?;
        // WAL is not available for in-memory databases; ignore the result there.
        let _ = conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()));
        conn.execute_batch(
            "PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS checkpoints (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 backend TEXT NOT NULL,
                 prefix_hash TEXT NOT NULL,
                 backend_version TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 pending_call_ids TEXT NOT NULL,
                 snapshot_path TEXT NOT NULL,
                 size_bytes INTEGER NOT NULL,
                 created_at INTEGER NOT NULL,
                 last_used_at INTEGER NOT NULL,
                 snapshot_json TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS checkpoints_key ON checkpoints(backend, prefix_hash);
             CREATE INDEX IF NOT EXISTS checkpoints_lru ON checkpoints(last_used_at);
             CREATE TABLE IF NOT EXISTS quota (
                 backend TEXT PRIMARY KEY,
                 status TEXT NOT NULL,
                 utilization REAL,
                 resets_at INTEGER,
                 backoff_s INTEGER NOT NULL DEFAULT 0,
                 updated_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS backend_state (
                 backend TEXT PRIMARY KEY,
                 mode TEXT NOT NULL,
                 cli_version TEXT NOT NULL,
                 self_test_at INTEGER NOT NULL,
                 self_test_result TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS request_log (
                 id TEXT PRIMARY KEY,
                 created_at INTEGER NOT NULL,
                 model TEXT NOT NULL,
                 backend TEXT,
                 checkpoint_hit_depth INTEGER,
                 appended_messages INTEGER,
                 finish_reason TEXT,
                 error_class TEXT,
                 tokens_in INTEGER NOT NULL,
                 tokens_out INTEGER NOT NULL,
                 cache_read INTEGER NOT NULL,
                 ttfb_ms INTEGER,
                 total_ms INTEGER NOT NULL
             );",
        )?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A poisoned lock only means another thread panicked mid-query; the connection
        // itself is still usable.
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ------------------------------------------------------------ checkpoints

    /// Index a new checkpoint. Never updates an existing row.
    pub fn insert_checkpoint(
        &self,
        backend: &str,
        prefix_hash: &str,
        snapshot: &SnapshotInfo,
        now_ms: i64,
    ) -> anyhow::Result<i64> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO checkpoints (backend, prefix_hash, backend_version, kind, pending_call_ids,
                 snapshot_path, size_bytes, created_at, last_used_at, snapshot_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9)",
            params![
                backend,
                prefix_hash,
                snapshot.backend_version,
                kind_str(snapshot.kind),
                serde_json::to_string(&snapshot.pending_call_ids)?,
                snapshot.path,
                to_i64(snapshot.size_bytes),
                now_ms,
                serde_json::to_string(snapshot)?,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// For each of `hashes` that has a checkpoint of `backend` written by `version`,
    /// the newest such checkpoint, keyed by depth (`hashes[k-1]` → depth `k`).
    pub fn lookup(
        &self,
        backend: &str,
        version: &str,
        hashes: &[String],
    ) -> anyhow::Result<BTreeMap<usize, CheckpointRecord>> {
        let mut out = BTreeMap::new();
        if hashes.is_empty() {
            return Ok(out);
        }
        let depth_of: std::collections::HashMap<&str, usize> =
            hashes.iter().enumerate().map(|(i, h)| (h.as_str(), i + 1)).collect();
        let conn = self.lock();
        // Chunk to stay well under SQLite's parameter limit.
        for chunk in hashes.chunks(500) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let sql = format!(
                "SELECT id, backend, prefix_hash, snapshot_json, created_at, last_used_at
                 FROM checkpoints WHERE backend = ? AND backend_version = ? AND prefix_hash IN ({placeholders})
                 ORDER BY id DESC"
            );
            let mut stmt = conn.prepare(&sql)?;
            let mut values: Vec<&dyn rusqlite::ToSql> = vec![&backend, &version];
            for h in chunk {
                values.push(h);
            }
            let rows = stmt.query_map(values.as_slice(), |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            })?;
            for row in rows {
                let (id, backend, prefix_hash, json, created_at, last_used_at) = row?;
                let Some(&depth) = depth_of.get(prefix_hash.as_str()) else { continue };
                if out.contains_key(&depth) {
                    continue; // newest already taken (ORDER BY id DESC)
                }
                let snapshot: SnapshotInfo = match serde_json::from_str(&json) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(id, error = %e, "unreadable checkpoint row ignored");
                        continue;
                    }
                };
                out.insert(depth, CheckpointRecord { id, backend, prefix_hash, snapshot, created_at, last_used_at });
            }
        }
        Ok(out)
    }

    pub fn touch_checkpoint(&self, id: i64, now_ms: i64) -> anyhow::Result<()> {
        self.lock().execute("UPDATE checkpoints SET last_used_at = ?1 WHERE id = ?2", params![now_ms, id])?;
        Ok(())
    }

    /// Remove one index row (e.g. its snapshot turned out to be missing).
    pub fn remove_checkpoint(&self, id: i64) -> anyhow::Result<Option<CheckpointRecord>> {
        let conn = self.lock();
        let rec = Self::get_locked(&conn, id)?;
        conn.execute("DELETE FROM checkpoints WHERE id = ?1", params![id])?;
        Ok(rec)
    }

    fn get_locked(conn: &Connection, id: i64) -> anyhow::Result<Option<CheckpointRecord>> {
        let row = conn
            .query_row(
                "SELECT id, backend, prefix_hash, snapshot_json, created_at, last_used_at FROM checkpoints WHERE id = ?1",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, i64>(5)?,
                    ))
                },
            )
            .optional()?;
        Ok(row.and_then(|(id, backend, prefix_hash, json, created_at, last_used_at)| {
            serde_json::from_str(&json)
                .ok()
                .map(|snapshot| CheckpointRecord { id, backend, prefix_hash, snapshot, created_at, last_used_at })
        }))
    }

    /// (count, total bytes)
    pub fn checkpoint_stats(&self) -> anyhow::Result<(u64, u64)> {
        let conn = self.lock();
        let (n, size): (i64, i64) = conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM checkpoints",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        Ok((to_u64(n), to_u64(size)))
    }

    /// Select and remove checkpoints that are expired (`last_used_at < now - ttl`) or,
    /// oldest-used first, exceed `max_bytes` in total. Snapshot paths in `in_use` are
    /// kept. Returns the removed records; the caller deletes the snapshot files.
    pub fn gc(
        &self,
        now_ms: i64,
        ttl_ms: i64,
        max_bytes: u64,
        in_use: &HashSet<String>,
    ) -> anyhow::Result<Vec<CheckpointRecord>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, backend, prefix_hash, snapshot_json, created_at, last_used_at, size_bytes, snapshot_path
             FROM checkpoints ORDER BY last_used_at ASC, id ASC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, i64>(6)?,
                    r.get::<_, String>(7)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);

        let mut total: u64 = rows.iter().map(|r| to_u64(r.6)).sum();
        let cutoff = now_ms.saturating_sub(ttl_ms);
        let mut removed = Vec::new();
        for (id, backend, prefix_hash, json, created_at, last_used_at, size, path) in rows {
            let expired = last_used_at < cutoff;
            let over = total > max_bytes;
            if !(expired || over) || in_use.contains(&path) {
                continue;
            }
            conn.execute("DELETE FROM checkpoints WHERE id = ?1", params![id])?;
            total = total.saturating_sub(to_u64(size));
            match serde_json::from_str::<SnapshotInfo>(&json) {
                Ok(snapshot) => {
                    removed.push(CheckpointRecord { id, backend, prefix_hash, snapshot, created_at, last_used_at })
                }
                Err(e) => tracing::warn!(id, error = %e, "unreadable checkpoint row removed"),
            }
        }
        Ok(removed)
    }

    // ------------------------------------------------------------ quota

    pub fn load_quota(&self) -> anyhow::Result<BTreeMap<String, QuotaRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT backend, status, utilization, resets_at, backoff_s FROM quota")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                QuotaRow {
                    status: r.get(1)?,
                    utilization: r.get(2)?,
                    resets_at: r.get::<_, Option<i64>>(3)?.map(to_u64),
                    backoff_s: to_u64(r.get(4)?),
                },
            ))
        })?;
        let mut out = BTreeMap::new();
        for row in rows {
            let (b, q) = row?;
            out.insert(b, q);
        }
        Ok(out)
    }

    pub fn save_quota(&self, backend: &str, row: &QuotaRow, now_s: u64) -> anyhow::Result<()> {
        self.lock().execute(
            "INSERT INTO quota (backend, status, utilization, resets_at, backoff_s, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(backend) DO UPDATE SET status = excluded.status, utilization = excluded.utilization,
                 resets_at = excluded.resets_at, backoff_s = excluded.backoff_s, updated_at = excluded.updated_at",
            params![backend, row.status, row.utilization, row.resets_at.map(to_i64), to_i64(row.backoff_s), to_i64(now_s)],
        )?;
        Ok(())
    }

    // ------------------------------------------------------------ backend state

    pub fn save_backend_state(&self, backend: &str, mode: &str, version: &str, now_s: u64, result: &str) -> anyhow::Result<()> {
        self.lock().execute(
            "INSERT INTO backend_state (backend, mode, cli_version, self_test_at, self_test_result)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(backend) DO UPDATE SET mode = excluded.mode, cli_version = excluded.cli_version,
                 self_test_at = excluded.self_test_at, self_test_result = excluded.self_test_result",
            params![backend, mode, version, to_i64(now_s), result],
        )?;
        Ok(())
    }

    // ------------------------------------------------------------ request log

    pub fn log_request(&self, row: &RequestLogRow, now_s: u64) -> anyhow::Result<()> {
        self.lock().execute(
            "INSERT INTO request_log (id, created_at, model, backend, checkpoint_hit_depth, appended_messages,
                 finish_reason, error_class, tokens_in, tokens_out, cache_read, ttfb_ms, total_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                row.id,
                to_i64(now_s),
                row.model,
                row.backend,
                row.checkpoint_hit_depth.map(to_i64),
                row.appended_messages.map(to_i64),
                row.finish_reason,
                row.error_class,
                to_i64(row.tokens_in),
                to_i64(row.tokens_out),
                to_i64(row.cache_read),
                row.ttfb_ms.map(to_i64),
                to_i64(row.total_ms),
            ],
        )?;
        Ok(())
    }

    /// Most recent request log rows (newest first). Used by tests and status.
    pub fn recent_requests(&self, limit: usize) -> anyhow::Result<Vec<RequestLogRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, model, backend, checkpoint_hit_depth, appended_messages, finish_reason, error_class,
                    tokens_in, tokens_out, cache_read, ttfb_ms, total_ms
             FROM request_log ORDER BY rowid DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![to_i64(limit as u64)], |r| {
            Ok(RequestLogRow {
                id: r.get(0)?,
                model: r.get(1)?,
                backend: r.get(2)?,
                checkpoint_hit_depth: r.get::<_, Option<i64>>(3)?.map(to_u64),
                appended_messages: r.get::<_, Option<i64>>(4)?.map(to_u64),
                finish_reason: r.get(5)?,
                error_class: r.get(6)?,
                tokens_in: to_u64(r.get(7)?),
                tokens_out: to_u64(r.get(8)?),
                cache_read: to_u64(r.get(9)?),
                ttfb_ms: r.get::<_, Option<i64>>(10)?.map(to_u64),
                total_ms: to_u64(r.get(11)?),
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(path: &str, kind: CheckpointKind, version: &str, size: u64) -> SnapshotInfo {
        SnapshotInfo {
            path: path.into(),
            kind,
            pending_call_ids: if kind == CheckpointKind::Awaiting { vec!["c1".into()] } else { vec![] },
            size_bytes: size,
            backend_version: version.into(),
        }
    }

    fn hashes(n: usize) -> Vec<String> {
        (1..=n).map(|i| format!("h{i}")).collect()
    }

    #[test]
    fn lookup_by_depth_backend_and_version() {
        let s = Store::open_in_memory().unwrap();
        s.insert_checkpoint("a", "h2", &snap("/p2", CheckpointKind::Idle, "v1", 10), 1).unwrap();
        s.insert_checkpoint("a", "h4", &snap("/p4", CheckpointKind::Awaiting, "v1", 10), 2).unwrap();
        s.insert_checkpoint("a", "h5", &snap("/p5", CheckpointKind::Idle, "v0", 10), 3).unwrap();
        s.insert_checkpoint("b", "h3", &snap("/b3", CheckpointKind::Idle, "v1", 10), 4).unwrap();

        let found = s.lookup("a", "v1", &hashes(6)).unwrap();
        assert_eq!(found.keys().copied().collect::<Vec<_>>(), vec![2, 4]);
        assert_eq!(found[&4].snapshot.kind, CheckpointKind::Awaiting);
        assert_eq!(found[&4].snapshot.pending_call_ids, vec!["c1".to_string()]);
        // Version filter: v0 snapshot only visible to v0.
        let old = s.lookup("a", "v0", &hashes(6)).unwrap();
        assert_eq!(old.keys().copied().collect::<Vec<_>>(), vec![5]);
        // Backend filter.
        let b = s.lookup("b", "v1", &hashes(6)).unwrap();
        assert_eq!(b.keys().copied().collect::<Vec<_>>(), vec![3]);
        // Shorter request does not see deeper checkpoints.
        let short = s.lookup("a", "v1", &hashes(3)).unwrap();
        assert_eq!(short.keys().copied().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn newest_checkpoint_wins_for_same_key() {
        let s = Store::open_in_memory().unwrap();
        s.insert_checkpoint("a", "h2", &snap("/old", CheckpointKind::Idle, "v1", 10), 1).unwrap();
        s.insert_checkpoint("a", "h2", &snap("/new", CheckpointKind::Idle, "v1", 10), 2).unwrap();
        let found = s.lookup("a", "v1", &hashes(2)).unwrap();
        assert_eq!(found[&2].snapshot.path, "/new");
        assert_eq!(s.checkpoint_stats().unwrap(), (2, 20));
    }

    #[test]
    fn many_hashes_are_chunked() {
        let s = Store::open_in_memory().unwrap();
        s.insert_checkpoint("a", "h1200", &snap("/deep", CheckpointKind::Idle, "v1", 1), 1).unwrap();
        let found = s.lookup("a", "v1", &hashes(1500)).unwrap();
        assert_eq!(found.keys().copied().collect::<Vec<_>>(), vec![1200]);
    }

    #[test]
    fn gc_ttl_then_lru_size_and_in_use() {
        let s = Store::open_in_memory().unwrap();
        // last_used: 100, 200, 300, 400
        for (i, t) in [100i64, 200, 300, 400].iter().enumerate() {
            s.insert_checkpoint("a", &format!("h{i}"), &snap(&format!("/p{i}"), CheckpointKind::Idle, "v1", 100), *t)
                .unwrap();
        }
        // Touch the oldest so it becomes newest.
        let found = s.lookup("a", "v1", &["h0".to_string()]).unwrap();
        s.touch_checkpoint(found[&1].id, 500).unwrap();

        // TTL: cutoff 1000-750 = 250 → p1 (200) expired; p0 was touched (500).
        let removed = s.gc(1000, 750, u64::MAX, &HashSet::new()).unwrap();
        assert_eq!(removed.iter().map(|r| r.snapshot.path.clone()).collect::<Vec<_>>(), vec!["/p1"]);
        // Size cap 150 bytes with 300 bytes left: LRU order p2(300), p3(400), p0(500).
        // p2 in use → skipped, so p3 and then (still over? 300-100=200 > 150) p0... check:
        let in_use: HashSet<String> = ["/p2".to_string()].into_iter().collect();
        let removed = s.gc(1000, 1_000_000, 150, &in_use).unwrap();
        assert_eq!(removed.iter().map(|r| r.snapshot.path.clone()).collect::<Vec<_>>(), vec!["/p3", "/p0"]);
        assert_eq!(s.checkpoint_stats().unwrap(), (1, 100));
    }

    #[test]
    fn quota_and_log_roundtrip() {
        let s = Store::open_in_memory().unwrap();
        let q = QuotaRow { status: "exhausted".into(), utilization: Some(1.0), resets_at: Some(123), backoff_s: 900 };
        s.save_quota("a", &q, 1).unwrap();
        s.save_quota("a", &q, 2).unwrap();
        assert_eq!(s.load_quota().unwrap()["a"], q);
        let row = RequestLogRow { id: "r1".into(), model: "m".into(), tokens_in: 5, total_ms: 3, ..Default::default() };
        s.log_request(&row, 1).unwrap();
        assert_eq!(s.recent_requests(10).unwrap(), vec![row]);
        s.save_backend_state("a", "native", "v1", 1, "ok").unwrap();
    }
}
