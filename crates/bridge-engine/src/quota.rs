//! Backend quota state machine (ARCHITECTURE.md section 10.1) and candidate ordering
//! (section 7 step 4).
//!
//! ```text
//! AVAILABLE ──warning──► WARNING ──ok──► AVAILABLE
//!     └──── limit error / exhausted event ────► EXHAUSTED (until resets_at or backoff)
//!                                                   │ time passes
//!                                                   ▼
//!                                  PROBING ──success──► AVAILABLE
//!                                     └──fail──► EXHAUSTED (backoff doubled)
//! ```
//! Pure functions take `now` (unix seconds) so they are testable without a clock.

use bridge_types::{EngineConfig, LimitInfo, LimitStatus};

/// Backoff used when a backend is exhausted and does not say when it resets
/// (`engine.quota_default_backoff_s` / `engine.quota_max_backoff_s`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backoff {
    /// First backoff after an exhaustion without a known reset time.
    pub default_s: u64,
    /// Upper bound for doubled backoffs.
    pub max_s: u64,
}

impl Backoff {
    pub fn from_config(cfg: &EngineConfig) -> Self {
        Backoff { default_s: cfg.quota_default_backoff_s, max_s: cfg.quota_max_backoff_s }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaStatus {
    Available,
    Warning,
    Exhausted,
    Probing,
}

impl QuotaStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            QuotaStatus::Available => "available",
            QuotaStatus::Warning => "warning",
            QuotaStatus::Exhausted => "exhausted",
            QuotaStatus::Probing => "probing",
        }
    }

    pub fn parse(s: &str) -> QuotaStatus {
        match s {
            "warning" => QuotaStatus::Warning,
            "exhausted" => QuotaStatus::Exhausted,
            "probing" => QuotaStatus::Probing,
            _ => QuotaStatus::Available,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuotaState {
    pub status: QuotaStatus,
    pub utilization: Option<f64>,
    /// Exhausted until this unix second.
    pub resets_at: Option<u64>,
    /// Backoff used for the last exhaustion without a known reset time.
    pub backoff_s: u64,
}

impl Default for QuotaState {
    fn default() -> Self {
        QuotaState { status: QuotaStatus::Available, utilization: None, resets_at: None, backoff_s: 0 }
    }
}

impl QuotaState {
    /// Apply the passage of time: an expired EXHAUSTED becomes PROBING.
    /// Returns true if the state changed.
    pub fn tick(&mut self, now: u64) -> bool {
        if self.status == QuotaStatus::Exhausted && self.resets_at.is_none_or(|r| now >= r) {
            self.status = QuotaStatus::Probing;
            return true;
        }
        false
    }

    /// Current status after `tick`.
    pub fn current(&mut self, now: u64) -> QuotaStatus {
        self.tick(now);
        self.status
    }

    /// Seconds until the backend may be retried (only meaningful when exhausted).
    pub fn retry_after(&self, now: u64) -> Option<u64> {
        match self.status {
            QuotaStatus::Exhausted => self.resets_at.map(|r| r.saturating_sub(now).max(1)),
            _ => None,
        }
    }

    fn next_backoff(&self, backoff: Backoff) -> u64 {
        if self.status == QuotaStatus::Probing && self.backoff_s > 0 {
            self.backoff_s.saturating_mul(2).min(backoff.max_s)
        } else {
            backoff.default_s
        }
    }

    fn exhaust(&mut self, resets_at: Option<u64>, now: u64, backoff: Backoff) {
        let until = match resets_at {
            Some(r) if r > now => r,
            _ => {
                let b = self.next_backoff(backoff);
                self.backoff_s = b;
                now.saturating_add(b)
            }
        };
        self.status = QuotaStatus::Exhausted;
        self.resets_at = Some(until);
    }

    /// A `Limit` event from the backend.
    pub fn on_limit(&mut self, info: &LimitInfo, now: u64, backoff: Backoff) {
        if info.utilization.is_some() {
            self.utilization = info.utilization;
        }
        match info.status {
            LimitStatus::Ok => {
                self.status = QuotaStatus::Available;
                self.resets_at = None;
                self.backoff_s = 0;
            }
            LimitStatus::Warning => {
                self.status = QuotaStatus::Warning;
                self.resets_at = None;
            }
            LimitStatus::Exhausted => self.exhaust(info.resets_at, now, backoff),
        }
    }

    /// A turn failed with `QuotaExhausted` (`retry_after` seconds, if known).
    pub fn on_quota_error(&mut self, retry_after: Option<u64>, now: u64, backoff: Backoff) {
        self.exhaust(retry_after.map(|s| now.saturating_add(s.max(1))), now, backoff);
    }

    /// A turn completed successfully.
    pub fn on_success(&mut self) {
        if self.status == QuotaStatus::Probing {
            self.status = QuotaStatus::Available;
            self.resets_at = None;
            self.backoff_s = 0;
        }
    }
}

/// Router input for one backend.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateInfo {
    pub name: String,
    pub priority: u32,
    pub status: QuotaStatus,
    /// Depth of the deepest usable checkpoint on this backend (0 = none).
    pub depth: usize,
}

/// Order candidates: the backend owning the deepest usable checkpoint first, then
/// AVAILABLE/PROBING by priority, then WARNING by priority. EXHAUSTED ones are dropped.
pub fn order_candidates(mut cands: Vec<CandidateInfo>) -> Vec<CandidateInfo> {
    cands.retain(|c| c.status != QuotaStatus::Exhausted);
    let group = |s: QuotaStatus| if s == QuotaStatus::Warning { 1 } else { 0 };
    cands.sort_by(|a, b| {
        group(a.status)
            .cmp(&group(b.status))
            .then(a.priority.cmp(&b.priority))
            .then(a.name.cmp(&b.name))
    });
    let best = cands.iter().enumerate().filter(|(_, c)| c.depth > 0).max_by(|(ia, a), (ib, b)| {
        // deepest; ties broken by earlier position
        a.depth.cmp(&b.depth).then(ib.cmp(ia))
    });
    if let Some((i, _)) = best {
        let c = cands.remove(i);
        cands.insert(0, c);
    }
    cands
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT_BACKOFF_S: u64 = 15 * 60;

    fn bo() -> Backoff {
        Backoff::from_config(&EngineConfig::default())
    }

    fn info(status: LimitStatus, resets_at: Option<u64>) -> LimitInfo {
        LimitInfo { status, utilization: None, resets_at }
    }

    #[test]
    fn exhausted_until_reset_then_probing_then_available() {
        let mut q = QuotaState::default();
        q.on_limit(&info(LimitStatus::Exhausted, Some(1_000)), 100, bo());
        assert_eq!(q.current(500), QuotaStatus::Exhausted);
        assert_eq!(q.retry_after(500), Some(500));
        assert_eq!(q.current(1_000), QuotaStatus::Probing);
        q.on_success();
        assert_eq!(q.status, QuotaStatus::Available);
    }

    #[test]
    fn unknown_reset_uses_default_backoff_and_doubles_after_failed_probe() {
        let mut q = QuotaState::default();
        q.on_quota_error(None, 0, bo());
        assert_eq!(q.status, QuotaStatus::Exhausted);
        assert_eq!(q.resets_at, Some(DEFAULT_BACKOFF_S));
        assert_eq!(q.current(DEFAULT_BACKOFF_S), QuotaStatus::Probing);
        q.on_quota_error(None, DEFAULT_BACKOFF_S, bo());
        assert_eq!(q.resets_at, Some(DEFAULT_BACKOFF_S + 2 * DEFAULT_BACKOFF_S));
        // Known retry_after wins.
        let mut q = QuotaState::default();
        q.on_quota_error(Some(30), 10, bo());
        assert_eq!(q.resets_at, Some(40));
        assert_eq!(q.retry_after(10), Some(30));
    }

    #[test]
    fn configured_backoff_is_used_and_capped() {
        let b = Backoff { default_s: 100, max_s: 250 };
        let mut q = QuotaState::default();
        q.on_quota_error(None, 0, b);
        assert_eq!(q.resets_at, Some(100));
        assert_eq!(q.current(100), QuotaStatus::Probing);
        q.on_quota_error(None, 100, b);
        assert_eq!(q.resets_at, Some(300)); // doubled
        assert_eq!(q.current(300), QuotaStatus::Probing);
        q.on_quota_error(None, 300, b);
        assert_eq!(q.resets_at, Some(550)); // capped at max_s
    }

    #[test]
    fn warning_and_ok() {
        let mut q = QuotaState::default();
        q.on_limit(&LimitInfo { status: LimitStatus::Warning, utilization: Some(0.9), resets_at: None }, 0, bo());
        assert_eq!(q.status, QuotaStatus::Warning);
        assert_eq!(q.utilization, Some(0.9));
        q.on_success(); // success alone does not clear a warning
        assert_eq!(q.status, QuotaStatus::Warning);
        q.on_limit(&info(LimitStatus::Ok, None), 0, bo());
        assert_eq!(q.status, QuotaStatus::Available);
        // Exhausted with a reset in the past falls back to backoff.
        q.on_limit(&info(LimitStatus::Exhausted, Some(5)), 10, bo());
        assert_eq!(q.resets_at, Some(10 + DEFAULT_BACKOFF_S));
    }

    fn c(name: &str, priority: u32, status: QuotaStatus, depth: usize) -> CandidateInfo {
        CandidateInfo { name: name.into(), priority, status, depth }
    }

    fn names(v: Vec<CandidateInfo>) -> Vec<String> {
        v.into_iter().map(|c| c.name).collect()
    }

    #[test]
    fn routing_order() {
        // Priority order, warning last, exhausted dropped.
        let v = order_candidates(vec![
            c("w", 0, QuotaStatus::Warning, 0),
            c("b", 2, QuotaStatus::Available, 0),
            c("a", 1, QuotaStatus::Available, 0),
            c("x", 0, QuotaStatus::Exhausted, 9),
            c("p", 3, QuotaStatus::Probing, 0),
        ]);
        assert_eq!(names(v), vec!["a", "b", "p", "w"]);
        // Checkpoint owner goes first, even if it is a WARNING backend.
        let v = order_candidates(vec![
            c("a", 1, QuotaStatus::Available, 2),
            c("b", 2, QuotaStatus::Available, 0),
            c("w", 0, QuotaStatus::Warning, 4),
        ]);
        assert_eq!(names(v), vec!["w", "a", "b"]);
        // Exhausted owner of a checkpoint is skipped.
        let v = order_candidates(vec![c("a", 1, QuotaStatus::Exhausted, 5), c("b", 2, QuotaStatus::Available, 0)]);
        assert_eq!(names(v), vec!["b"]);
    }
}
