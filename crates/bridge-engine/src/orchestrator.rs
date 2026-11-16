//! The single request path (ARCHITECTURE.md section 7).
//!
//! Structure of one `chat` call:
//!
//! ```text
//! chat() ─► coordinator task ─► prepare (route, hashes, checkpoint lookups, candidates)
//!                │                 └─ error → Error{index 0}, end
//!                ├─ choice task 0 ─┐  each: candidates in order → permit → plan → attempts
//!                ├─ choice task 1 ─┤        (policy retries) → events / Failed → freeze/discard
//!                │                 ▼
//!                └─ gate: buffer events until EVERY choice has emitted its first event.
//!                   Any choice failing (or erroring) before that → the whole request fails:
//!                   the only item sent is Error{index 0}; other choices are cancelled.
//!                   Once open, events are forwarded as they come; later errors are
//!                   per-choice Error events (mid-stream errors).
//! ```
//!
//! For n = 1 the gate opens on the first event of choice 0, so streaming is never
//! delayed. For n > 1, choice 0's first chunk waits until all choices have started,
//! which is what guarantees "first item is the error" when any choice fails early.
//!
//! The returned stream holds a drop guard on a child cancellation token: if the
//! consumer drops it (client gone), every backend turn is aborted and discarded.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bridge_policy::{system_addendum, OutputPolicy, PolicyOutput, PolicySpec};
use bridge_types::{
    BackendMode, BridgeError, CancellationToken, ChatRequest, CheckpointKind, CheckpointRef, ChoiceEvent, ErrorClass,
    FinishReason, ModelRoute, ResponseFormat, Turn, TurnEvent, TurnSpec, Usage,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use tokio::sync::{mpsc, OwnedSemaphorePermit};

use crate::norm::{next_hash, prefix_hashes};
use crate::pending::{PendingGuard, PendingMatch};
use crate::planner::{choose_base, compatible, is_side_request, leading_system_text, split_suffix};
use crate::quota::{order_candidates, CandidateInfo, QuotaStatus};
use crate::store::{CheckpointRecord, RequestLogRow};
use crate::{now_ms, now_s, Inner, Slot};

// ---------------------------------------------------------------- public errors

/// Caller-safe version of a backend error: fixed text per class (ERR-11, LEAK-3).
/// The raw error is logged by the caller of this function.
pub(crate) fn public_error(e: &BridgeError) -> BridgeError {
    let message = match e.class {
        // Validation-type errors carry caller-facing text by contract.
        ErrorClass::InvalidRequest
        | ErrorClass::Unauthorized
        | ErrorClass::ModelNotFound
        | ErrorClass::NotFound
        | ErrorClass::PayloadTooLarge => return e.clone(),
        ErrorClass::ContextTooLong => {
            "This model's maximum context length was exceeded. Please reduce the length of the messages."
        }
        ErrorClass::QuotaExhausted => "You exceeded your current quota. Please retry later.",
        ErrorClass::RateLimited => "Too many concurrent requests. Please retry shortly.",
        ErrorClass::Unavailable => "The model is currently overloaded or unavailable. Please retry shortly.",
        ErrorClass::BackendAuth => "The model is currently unavailable. Please retry later.",
        ErrorClass::Timeout => "The model did not respond in time. Please retry.",
        ErrorClass::Internal => "The server had an error while processing your request. Please retry.",
    };
    BridgeError { class: e.class, message: message.to_string(), param: None, retry_after: e.retry_after }
}

// ---------------------------------------------------------------- request context

struct Candidate {
    name: String,
    mode: BackendMode,
    lookups: BTreeMap<usize, CheckpointRecord>,
}

struct RequestCtx {
    request: ChatRequest,
    route: ModelRoute,
    hashes: Vec<String>,
    side: bool,
    policy_spec: PolicySpec,
    system: String,
    candidates: Vec<Candidate>,
    /// All backends the route could use (for Retry-After computation).
    route_backends: Vec<String>,
    started: Instant,
}

/// One backend the route may use, while candidates are being collected.
struct Probe {
    name: String,
    mode: BackendMode,
    priority: u32,
    status: QuotaStatus,
    version: String,
    lookups: BTreeMap<usize, CheckpointRecord>,
    /// A promised checkpoint deeper than every usable stored one (17.1).
    pending: Option<PendingMatch>,
}

/// Build the request context, or the error to return before anything runs.
/// `Ok(None)`: the request was cancelled while waiting for a pending checkpoint.
async fn prepare(
    inner: &Arc<Inner>,
    request: ChatRequest,
    started: Instant,
    cancel: &CancellationToken,
) -> Result<Option<RequestCtx>, BridgeError> {
    let Some(route) = inner.config.models.get(&request.model).cloned() else {
        return Err(BridgeError {
            class: ErrorClass::ModelNotFound,
            message: format!("The model `{}` does not exist or you do not have access to it.", request.model),
            param: Some("model".into()),
            retry_after: None,
        });
    };
    if request.messages.is_empty() {
        return Err(BridgeError::invalid("'messages' must contain at least one message.", Some("messages")));
    }
    let max_n = inner.config.output.max_n.max(1);
    if request.n > max_n {
        return Err(BridgeError::invalid(format!("'n' must be at most {max_n}."), Some("n")));
    }

    let hashes = prefix_hashes(&request.messages);
    let side = is_side_request(&request.messages, request.tools.len());
    let policy_spec = PolicySpec {
        chars_per_token: inner.config.engine.chars_per_token,
        json_max_candidates: inner.config.output.json_extract_max_candidates,
        instructions: inner.config.output.instructions.clone(),
        ..PolicySpec::from_request(&request)
    };
    let mut system = leading_system_text(&request.messages);
    if let Some(add) = system_addendum(&policy_spec) {
        if !add.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&add);
        }
    }

    let route_backends: Vec<String> = match &route.backend {
        Some(b) => vec![b.clone()],
        None => inner.slots.keys().cloned().collect(),
    };

    let mut probes: Vec<Probe> = Vec::new();
    let mut blocked_by_quota = false;
    for name in &route_backends {
        let Some(slot) = inner.slots.get(name) else {
            tracing::error!(backend = %name, model = %request.model, "model routed to an unknown backend");
            continue;
        };
        if !slot.cfg.enabled {
            continue;
        }
        let mode = slot.backend.mode();
        if mode == BackendMode::Disabled || inner.in_auth_cooldown(name) {
            continue;
        }
        let status = inner.quota_status(name);
        if status == QuotaStatus::Exhausted {
            blocked_by_quota = true;
            continue;
        }
        let version = slot.backend.version();
        let (lookups, pending) = if mode == BackendMode::Native {
            // Promises are read before the index: a promise resolved in between is then
            // already in the index (records are inserted before their promise resolves).
            let messages = &request.messages;
            let pending = inner.pending.find_deepest(name, &hashes, |depth, kind, ids| {
                depth < messages.len() && compatible(kind, ids, &messages[depth..])
            });
            let lookups = match inner.store.lookup(name, &version, &hashes) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(backend = %name, error = %e, "checkpoint lookup failed; planning without");
                    BTreeMap::new()
                }
            };
            (lookups, pending)
        } else {
            (BTreeMap::new(), None)
        };
        let best = choose_base(&request.messages, &lookups).map(|(d, _)| d).unwrap_or(0);
        let pending = pending.filter(|p| p.depth > best);
        probes.push(Probe {
            name: name.clone(),
            mode,
            priority: slot.cfg.priority,
            status,
            version,
            lookups,
            pending,
        });
    }

    if !wait_for_pending(inner, &mut probes, cancel).await {
        return Ok(None);
    }

    // Routing depth is computed after the wait, so a promise that resolved counts in
    // full; one that failed or timed out cannot be used as a base and does not count.
    let infos = probes
        .iter()
        .map(|p| CandidateInfo {
            name: p.name.clone(),
            priority: p.priority,
            status: p.status,
            depth: choose_base(&request.messages, &p.lookups).map(|(d, _)| d).unwrap_or(0),
        })
        .collect();
    let mut lookups_by_name: BTreeMap<String, (BackendMode, BTreeMap<usize, CheckpointRecord>)> =
        probes.into_iter().map(|p| (p.name, (p.mode, p.lookups))).collect();

    let ordered = order_candidates(infos);
    if ordered.is_empty() {
        return Err(if blocked_by_quota {
            let e = BridgeError::new(ErrorClass::QuotaExhausted, "");
            let e = public_error(&e);
            match inner.quota_retry_after(&route_backends) {
                Some(s) => e.with_retry_after(s),
                None => e,
            }
        } else {
            public_error(&BridgeError::new(ErrorClass::Unavailable, ""))
        });
    }
    let candidates = ordered
        .into_iter()
        .filter_map(|c| {
            lookups_by_name.remove(&c.name).map(|(mode, lookups)| Candidate { name: c.name, mode, lookups })
        })
        .collect();

    Ok(Some(RequestCtx { request, route, hashes, side, policy_spec, system, candidates, route_backends, started }))
}

/// Wait (bounded by `engine.pending_wait_ms` and `cancel`) for the promised checkpoints
/// found by `prepare`, adding the ones that resolve to their backend's lookups.
/// Runs before any concurrency permit is taken. Returns false if cancelled.
async fn wait_for_pending(inner: &Inner, probes: &mut [Probe], cancel: &CancellationToken) -> bool {
    let budget = Duration::from_millis(inner.config.engine.pending_wait_ms);
    let waits: Vec<_> = probes
        .iter_mut()
        .enumerate()
        .filter_map(|(i, p)| p.pending.take().map(|m| (i, m)))
        .collect();
    if waits.is_empty() || budget.is_zero() {
        return true;
    }
    let started = Instant::now();
    let futs = waits.into_iter().map(|(i, m)| {
        tracing::debug!(depth = m.depth, kind = ?m.kind, calls = m.pending_call_ids.len(), "waiting for a pending checkpoint");
        let depth = m.depth;
        async move { (i, depth, tokio::time::timeout(budget, m.wait()).await) }
    });
    let results = tokio::select! {
        _ = cancel.cancelled() => {
            tracing::debug!("request cancelled while waiting for a pending checkpoint");
            return false;
        }
        r = futures::future::join_all(futs) => r,
    };
    let ms = started.elapsed().as_millis() as u64;
    for (i, depth, result) in results {
        let Some(p) = probes.get_mut(i) else { continue };
        match result {
            Ok(Some(rec)) if rec.backend == p.name && rec.snapshot.backend_version == p.version => {
                tracing::debug!(backend = %p.name, depth, ms, "pending checkpoint ready");
                p.lookups.insert(depth, rec);
            }
            Ok(Some(_)) => tracing::debug!(backend = %p.name, depth, ms, "pending checkpoint from another version"),
            Ok(None) => tracing::info!(backend = %p.name, depth, ms, "pending checkpoint failed; planning from an older base"),
            Err(_) => tracing::info!(backend = %p.name, depth, ms, "pending checkpoint not ready in time; planning from an older base"),
        }
    }
    true
}

/// Rough token counts (`engine.chars_per_token` characters per token) for a turn that ended
/// before the backend reported usage.
fn estimate_usage(
    system: &str,
    request: &ChatRequest,
    produced: &bridge_types::Message,
    chars_per_token: u32,
) -> Usage {
    fn chars(m: &bridge_types::Message) -> usize {
        use bridge_types::Part;
        m.parts
            .iter()
            .map(|p| match p {
                Part::Text { text } | Part::Reasoning { text } => text.chars().count(),
                Part::ToolCall(c) => c.name.len() + c.arguments.chars().count(),
                Part::ToolResult(r) => r.content.chars().count(),
                // Fixed rough per-image weight; images have no character length.
                Part::Image(_) => 3000,
            })
            .sum()
    }
    let tools: usize = request
        .tools
        .iter()
        .map(|t| t.name.len() + t.description.chars().count() + t.parameters.to_string().len())
        .sum();
    let prompt = system.chars().count() + tools + request.messages.iter().map(chars).sum::<usize>();
    let per_token = (chars_per_token as usize).max(1);
    Usage {
        prompt_tokens: (prompt / per_token).max(1) as u64,
        completion_tokens: (chars(produced) / per_token).max(1) as u64,
        cached_tokens: 0,
        reasoning_tokens: 0,
    }
}

// ---------------------------------------------------------------- per-choice plumbing

enum Msg {
    Event(u32, TurnEvent),
    /// The choice failed before emitting anything (error already caller-safe).
    Failed(u32, BridgeError),
}

struct Sink {
    index: u32,
    tx: mpsc::Sender<Msg>,
    emitted: bool,
    first_emit: Option<Instant>,
}

impl Sink {
    /// Forward an event; false if the consumer is gone.
    async fn emit(&mut self, ev: TurnEvent) -> bool {
        self.emitted = true;
        if self.first_emit.is_none() {
            self.first_emit = Some(Instant::now());
        }
        self.tx.send(Msg::Event(self.index, ev)).await.is_ok()
    }
}

#[derive(Default)]
struct ChoiceSummary {
    backend: Option<String>,
    depth: Option<u64>,
    appended: Option<u64>,
    finish: Option<FinishReason>,
    error: Option<ErrorClass>,
    usage: Usage,
    first_emit: Option<Instant>,
}

enum Attempt {
    Completed { finish: FinishReason, usage: Usage },
    Retry,
    /// Failed before anything reached the caller (raw error).
    StartFailed(BridgeError),
    Failed(BridgeError),
    /// Failed after output; the Error event was already emitted.
    FailedAfterOutput(ErrorClass),
    Cancelled,
}

/// Decrements `in_flight` and releases the base checkpoint when dropped.
struct RunGuard<'a> {
    inner: &'a Inner,
    slot: &'a Slot,
    base_path: Option<String>,
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.slot.in_flight.fetch_sub(1, Ordering::SeqCst);
        if let Some(p) = &self.base_path {
            self.inner.unmark_in_use(p);
        }
    }
}

fn label(class: ErrorClass) -> String {
    serde_json::to_value(class).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
}

fn finish_label(f: FinishReason) -> String {
    serde_json::to_value(f).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
}

// ---------------------------------------------------------------- entry point

pub(crate) fn chat(inner: Arc<Inner>, request: ChatRequest, cancel: CancellationToken) -> BoxStream<'static, ChoiceEvent> {
    let child = cancel.child_token();
    let guard = child.clone().drop_guard();
    let (out_tx, mut out_rx) = mpsc::channel::<ChoiceEvent>(256);
    tokio::spawn(coordinate(inner, request, child, out_tx));
    Box::pin(async_stream::stream! {
        let _guard = guard;
        while let Some(ev) = out_rx.recv().await {
            yield ev;
        }
    })
}

async fn coordinate(inner: Arc<Inner>, request: ChatRequest, cancel: CancellationToken, out_tx: mpsc::Sender<ChoiceEvent>) {
    let started = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();
    let model = request.model.clone();
    inner.maybe_recheck_versions();

    let ctx = match prepare(&inner, request, started, &cancel).await {
        Ok(Some(c)) => Arc::new(c),
        Ok(None) => {
            let row = RequestLogRow {
                id: request_id,
                model,
                error_class: Some("cancelled".into()),
                total_ms: started.elapsed().as_millis() as u64,
                ..Default::default()
            };
            drop(out_tx);
            log_request(&inner, &row);
            return;
        }
        Err(e) => {
            tracing::info!(request = %request_id, class = ?e.class, "request rejected before running");
            let row = RequestLogRow {
                id: request_id,
                model,
                error_class: Some(label(e.class)),
                total_ms: started.elapsed().as_millis() as u64,
                ..Default::default()
            };
            let _ = out_tx.send(ChoiceEvent { index: 0, event: TurnEvent::Error { error: e } }).await;
            drop(out_tx);
            log_request(&inner, &row);
            return;
        }
    };

    let n = ctx.request.n.max(1);
    let (itx, mut irx) = mpsc::channel::<Msg>(256);
    let handles: Vec<_> = (0..n)
        .map(|i| {
            let sink = Sink { index: i, tx: itx.clone(), emitted: false, first_emit: None };
            tokio::spawn(run_choice(inner.clone(), ctx.clone(), sink, cancel.clone()))
        })
        .collect();
    drop(itx);

    // Gate (see module docs).
    let mut out = Some(out_tx);
    let mut started_flags = vec![false; n as usize];
    let mut buffer: Vec<ChoiceEvent> = Vec::new();
    let mut gate_open = false;
    let mut request_error: Option<ErrorClass> = None;
    while let Some(msg) = irx.recv().await {
        let Some(tx) = out.as_ref() else { continue };
        let fail_early = match &msg {
            Msg::Failed(_, e) if !gate_open => Some(e.clone()),
            Msg::Event(_, TurnEvent::Error { error }) if !gate_open => Some(error.clone()),
            _ => None,
        };
        if let Some(err) = fail_early {
            request_error = Some(err.class);
            let _ = tx.send(ChoiceEvent { index: 0, event: TurnEvent::Error { error: err } }).await;
            out = None;
            cancel.cancel();
            continue;
        }
        let ev = match msg {
            Msg::Event(i, event) => ChoiceEvent { index: i, event },
            Msg::Failed(i, error) => ChoiceEvent { index: i, event: TurnEvent::Error { error } },
        };
        if gate_open {
            if tx.send(ev).await.is_err() {
                cancel.cancel();
                out = None;
            }
            continue;
        }
        if let Some(f) = started_flags.get_mut(ev.index as usize) {
            *f = true;
        }
        buffer.push(ev);
        if started_flags.iter().all(|f| *f) {
            gate_open = true;
            for ev in buffer.drain(..) {
                if tx.send(ev).await.is_err() {
                    cancel.cancel();
                    out = None;
                    break;
                }
            }
        }
    }
    drop(out);

    // Collect summaries and write the request log.
    let mut summaries = Vec::new();
    for h in handles {
        match h.await {
            Ok(s) => summaries.push(s),
            Err(e) => tracing::error!(error = %e, "choice task failed"),
        }
    }
    let first = summaries.first();
    let usage = summaries.iter().fold(Usage::default(), |mut acc, s| {
        acc.prompt_tokens += s.usage.prompt_tokens;
        acc.completion_tokens += s.usage.completion_tokens;
        acc.cached_tokens += s.usage.cached_tokens;
        acc
    });
    let error_class = request_error
        .or_else(|| summaries.iter().find_map(|s| s.error))
        .map(label)
        .or_else(|| if cancel.is_cancelled() && summaries.iter().any(|s| s.finish.is_none()) { Some("cancelled".into()) } else { None });
    let row = RequestLogRow {
        id: request_id,
        model,
        backend: first.and_then(|s| s.backend.clone()),
        checkpoint_hit_depth: first.and_then(|s| s.depth),
        appended_messages: first.and_then(|s| s.appended),
        finish_reason: first.and_then(|s| s.finish).map(finish_label),
        error_class,
        tokens_in: usage.prompt_tokens,
        tokens_out: usage.completion_tokens,
        cache_read: usage.cached_tokens,
        ttfb_ms: summaries
            .iter()
            .filter_map(|s| s.first_emit)
            .min()
            .map(|t| t.duration_since(ctx.started).as_millis() as u64),
        total_ms: started.elapsed().as_millis() as u64,
    };
    log_request(&inner, &row);
    inner.maybe_gc(false).await;
}

fn log_request(inner: &Inner, row: &RequestLogRow) {
    if let Err(e) = inner.store.log_request(row, now_s()) {
        tracing::warn!(error = %e, "writing request log failed");
    }
}

// ---------------------------------------------------------------- one choice

async fn acquire(slot: &Slot, wait: bool, cancel: &CancellationToken) -> Option<OwnedSemaphorePermit> {
    if let Ok(p) = slot.sem.clone().try_acquire_owned() {
        return Some(p);
    }
    if !wait {
        return None;
    }
    let wait_for = Duration::from_millis(slot.cfg.queue_wait_ms);
    tokio::select! {
        _ = cancel.cancelled() => None,
        r = tokio::time::timeout(wait_for, slot.sem.clone().acquire_owned()) => r.ok().and_then(Result::ok),
    }
}

async fn run_choice(inner: Arc<Inner>, ctx: Arc<RequestCtx>, mut sink: Sink, cancel: CancellationToken) -> ChoiceSummary {
    let mut summary = ChoiceSummary::default();
    let mut last_err: Option<BridgeError> = None;
    let total = ctx.candidates.len();

    'candidates: for (ci, cand) in ctx.candidates.iter().enumerate() {
        if cancel.is_cancelled() {
            return summary;
        }
        let Some(slot) = inner.slots.get(&cand.name) else { continue };
        if ci > 0 && inner.quota_status(&cand.name) == QuotaStatus::Exhausted {
            continue;
        }
        // Prefer a backend with a free slot; only the last candidate waits for one.
        let Some(permit) = acquire(slot, ci + 1 == total, &cancel).await else {
            if cancel.is_cancelled() {
                return summary;
            }
            last_err = Some(BridgeError::new(ErrorClass::RateLimited, "no concurrency slot").with_retry_after(1));
            continue;
        };

        // Held across policy re-runs; `run_attempt` releases it before resolving its
        // checkpoint promise so a waiter can take the slot right away.
        let mut permit = Some(permit);
        let mut lookups = cand.lookups.clone();
        let attempts = inner.config.output.enforcement_retries.saturating_add(1);
        let mut attempt = 0;
        loop {
            let base = choose_base(&ctx.request.messages, &lookups).map(|(d, r)| (d, r.clone()));
            let (append, trigger) = split_suffix(&ctx.request.messages, base.as_ref().map(|(d, _)| *d).unwrap_or(0));
            let model = match &ctx.route.model {
                Some(m) => m.clone(),
                None if ctx.side => slot.cfg.side_request_model.clone().unwrap_or_else(|| slot.cfg.default_model.clone()),
                None => slot.cfg.default_model.clone(),
            };
            let caps = slot.backend.capabilities();
            let json_schema = match &ctx.request.response_format {
                ResponseFormat::JsonSchema { schema, .. } if caps.json_schema => Some(schema.clone()),
                _ => None,
            };
            let spec = TurnSpec {
                model,
                system: ctx.system.clone(),
                tools: ctx.request.tools.clone(),
                base: base.as_ref().map(|(_, r)| CheckpointRef { backend: cand.name.clone(), snapshot: r.snapshot.clone() }),
                append,
                trigger,
                effort: ctx.request.reasoning_effort,
                json_schema,
                ephemeral: ctx.side,
            };
            summary.backend = Some(cand.name.clone());
            summary.depth = Some(base.as_ref().map(|(d, _)| *d as u64).unwrap_or(0));
            summary.appended = Some(spec.append.len() as u64);
            if let Some((_, rec)) = &base {
                if let Err(e) = inner.store.touch_checkpoint(rec.id, now_ms()) {
                    tracing::warn!(error = %e, "touching checkpoint failed");
                }
            }
            tracing::debug!(
                backend = %cand.name, choice = sink.index, depth = summary.depth, appended = summary.appended,
                ephemeral = spec.ephemeral, attempt, "running turn"
            );

            let final_attempt = attempt + 1 >= attempts;
            let outcome =
                run_attempt(&inner, &ctx, slot, cand.mode, spec, final_attempt, &mut sink, &cancel, &mut permit).await;
            match outcome {
                Attempt::Completed { finish, usage } => {
                    summary.finish = Some(finish);
                    summary.usage = usage;
                    summary.first_emit = sink.first_emit;
                    return summary;
                }
                Attempt::Retry => {
                    attempt += 1;
                    if attempt >= attempts.saturating_add(1) {
                        // The policy must not ask for a retry on the final attempt; be safe anyway.
                        let err = public_error(&BridgeError::new(ErrorClass::Internal, ""));
                        summary.error = Some(err.class);
                        if sink.emitted {
                            let _ = sink.emit(TurnEvent::Error { error: err }).await;
                        } else {
                            let _ = sink.tx.send(Msg::Failed(sink.index, err)).await;
                        }
                        return summary;
                    }
                    continue;
                }
                Attempt::Cancelled => {
                    summary.first_emit = sink.first_emit;
                    return summary;
                }
                Attempt::FailedAfterOutput(class) => {
                    summary.error = Some(class);
                    summary.first_emit = sink.first_emit;
                    return summary;
                }
                Attempt::StartFailed(e) | Attempt::Failed(e) => {
                    tracing::warn!(backend = %cand.name, choice = sink.index, class = ?e.class, raw = %e.message, "turn failed before output");
                    inner.on_failure(&cand.name, &e);
                    // A broken base (e.g. snapshot file gone) is a cache miss, not an error:
                    // drop it from the index and rebuild from the request on the same backend.
                    if let Some((depth, rec)) = &base {
                        if e.class == ErrorClass::Internal {
                            tracing::warn!(backend = %cand.name, "base checkpoint unusable; retrying without it");
                            let _ = inner.store.remove_checkpoint(rec.id);
                            lookups.remove(depth);
                            continue;
                        }
                    }
                    if e.is_failover_candidate() {
                        last_err = Some(e);
                        continue 'candidates;
                    }
                    summary.error = Some(e.class);
                    let _ = sink.tx.send(Msg::Failed(sink.index, public_error(&e))).await;
                    return summary;
                }
            }
        }
    }

    // Every candidate failed or was skipped.
    let err = match last_err {
        Some(e) if e.class == ErrorClass::QuotaExhausted || e.class == ErrorClass::RateLimited => {
            let mut p = public_error(&e);
            if e.class == ErrorClass::QuotaExhausted {
                if let Some(s) = inner.quota_retry_after(&ctx.route_backends) {
                    p.retry_after = Some(s);
                }
            }
            p
        }
        Some(e) => public_error(&e),
        None => {
            let p = public_error(&BridgeError::new(ErrorClass::QuotaExhausted, ""));
            match inner.quota_retry_after(&ctx.route_backends) {
                Some(s) => p.with_retry_after(s),
                None => p,
            }
        }
    };
    summary.error = Some(err.class);
    let _ = sink.tx.send(Msg::Failed(sink.index, err)).await;
    summary
}

enum Next {
    Cancel,
    Timeout,
    Event(Option<TurnEvent>),
}

#[allow(clippy::too_many_arguments)]
async fn run_attempt(
    inner: &Arc<Inner>,
    ctx: &RequestCtx,
    slot: &Slot,
    mode: BackendMode,
    spec: TurnSpec,
    final_attempt: bool,
    sink: &mut Sink,
    cancel: &CancellationToken,
    permit: &mut Option<OwnedSemaphorePermit>,
) -> Attempt {
    let name = slot.backend.name().to_string();
    let ephemeral = spec.ephemeral;
    let base_path = spec.base.as_ref().map(|b| b.snapshot.path.clone());
    if let Some(p) = &base_path {
        inner.mark_in_use(p);
    }
    slot.in_flight.fetch_add(1, Ordering::SeqCst);
    let _guard = RunGuard { inner, slot, base_path };

    let mut policy = OutputPolicy::new(ctx.policy_spec.clone(), final_attempt);
    let started = tokio::select! {
        _ = cancel.cancelled() => return Attempt::Cancelled,
        r = slot.backend.start_turn(spec) => r,
    };
    let Turn { mut events, mut control } = match started {
        Ok(t) => t,
        Err(e) => return Attempt::StartFailed(e),
    };

    let first_timeout = Duration::from_secs(inner.config.server.first_token_timeout_s.max(1));
    let drain_after_done = Duration::from_millis(inner.config.engine.drain_after_done_ms);
    let reasoning_ok = inner.config.output.reasoning_to_caller;
    let mut got_event = false;
    let mut backend_done = false;
    let mut stopped = false;
    let mut emitted_done: Option<FinishReason> = None;
    let mut usage = Usage::default();
    let mut usage_seen = false;
    // Promise of this turn's checkpoint (17.1): registered right before Done goes out,
    // resolved after freeze + index; dropping it resolves it as Failed.
    let mut promise: Option<(String, PendingGuard)> = None;

    // Applies policy outputs; returns Some(outcome) when the attempt must end now.
    macro_rules! apply {
        ($outs:expr) => {
            for out in $outs {
                match out {
                    PolicyOutput::Emit(TurnEvent::Limit { info }) => inner.on_limit(&name, &info),
                    PolicyOutput::Emit(TurnEvent::ReasoningDelta { .. }) if !reasoning_ok => {}
                    PolicyOutput::Emit(ev) => {
                        // The policy may end the turn (stop sequence, max_tokens) before the
                        // backend reported usage; the caller still gets plausible numbers.
                        if matches!(ev, TurnEvent::Done { .. }) && !usage_seen {
                            usage_seen = true;
                            let est = estimate_usage(
                                &ctx.system,
                                &ctx.request,
                                &policy.assistant_message(),
                                inner.config.engine.chars_per_token,
                            );
                            usage = est;
                            if !sink.emit(TurnEvent::Usage { usage: est }).await {
                                control.abort().await;
                                control.discard().await;
                                return Attempt::Cancelled;
                            }
                        }
                        match &ev {
                            TurnEvent::Usage { usage: u } => {
                                usage = *u;
                                usage_seen = true;
                            }
                            TurnEvent::Done { finish } => {
                                emitted_done = Some(*finish);
                                let publishable = mode == BackendMode::Native
                                    && !ephemeral
                                    && backend_done
                                    && !stopped
                                    && !policy.altered();
                                if publishable && promise.is_none() {
                                    let assistant = policy.assistant_message();
                                    let key = next_hash(ctx.hashes.last().map(String::as_str), &assistant);
                                    let ids: Vec<String> =
                                        assistant.tool_calls().iter().map(|c| c.id.clone()).collect();
                                    let kind =
                                        if ids.is_empty() { CheckpointKind::Idle } else { CheckpointKind::Awaiting };
                                    let guard = inner.pending.register(&name, &key, kind, ids);
                                    promise = Some((key, guard));
                                }
                            }
                            _ => {}
                        }
                        if !sink.emit(ev).await {
                            control.abort().await;
                            control.discard().await;
                            return Attempt::Cancelled;
                        }
                    }
                    PolicyOutput::StopBackend => stopped = true,
                    PolicyOutput::Retry { reason } => {
                        tracing::info!(backend = %name, choice = sink.index, %reason, "output policy requested a re-run");
                        control.abort().await;
                        control.discard().await;
                        return Attempt::Retry;
                    }
                }
            }
        };
    }

    while !stopped {
        // Once Done went out the caller has its complete answer and may hang up at any
        // moment (the frontend drops the stream after the last Done, which fires the
        // cancel token). From here on, ignore cancellation and just drain the backend so
        // the turn can still be frozen as a checkpoint (bounded by
        // `engine.drain_after_done_ms`).
        let next = if emitted_done.is_some() {
            match tokio::time::timeout(drain_after_done, events.next()).await {
                Ok(ev) => Next::Event(ev),
                Err(_) => Next::Timeout,
            }
        } else if got_event {
            tokio::select! {
                _ = cancel.cancelled() => Next::Cancel,
                ev = events.next() => Next::Event(ev),
            }
        } else {
            tokio::select! {
                _ = cancel.cancelled() => Next::Cancel,
                r = tokio::time::timeout(first_timeout, events.next()) => match r {
                    Ok(ev) => Next::Event(ev),
                    Err(_) => Next::Timeout,
                },
            }
        };
        let ev = match next {
            Next::Cancel => {
                control.abort().await;
                control.discard().await;
                return Attempt::Cancelled;
            }
            Next::Timeout => {
                control.abort().await;
                control.discard().await;
                if let Some(finish) = emitted_done {
                    tracing::warn!(backend = %name, choice = sink.index, "backend did not close after Done; not checkpointed");
                    return Attempt::Completed { finish, usage };
                }
                return Attempt::Failed(BridgeError::new(ErrorClass::Timeout, "no first event within timeout"));
            }
            Next::Event(None) => break,
            Next::Event(Some(ev)) => ev,
        };
        got_event = true;
        match ev {
            TurnEvent::Limit { info } => inner.on_limit(&name, &info),
            TurnEvent::Error { error } => {
                control.abort().await;
                control.discard().await;
                if !sink.emitted {
                    return Attempt::Failed(error);
                }
                tracing::warn!(backend = %name, choice = sink.index, class = ?error.class, raw = %error.message, "turn failed after output");
                inner.on_failure(&name, &error);
                let class = error.class;
                let _ = sink.emit(TurnEvent::Error { error: public_error(&error) }).await;
                return Attempt::FailedAfterOutput(class);
            }
            other => {
                if matches!(other, TurnEvent::Done { .. }) {
                    backend_done = true;
                }
                let outs = policy.on_event(other);
                apply!(outs);
            }
        }
    }
    drop(events);

    if stopped {
        control.abort().await;
    } else {
        let outs = policy.finish();
        apply!(outs);
    }

    let Some(finish) = emitted_done else {
        control.abort().await;
        control.discard().await;
        let err = BridgeError::new(ErrorClass::Internal, "turn ended without completion");
        if !sink.emitted {
            return Attempt::Failed(err);
        }
        tracing::warn!(backend = %name, choice = sink.index, "turn ended without Done after output");
        let _ = sink.emit(TurnEvent::Error { error: public_error(&err) }).await;
        return Attempt::FailedAfterOutput(ErrorClass::Internal);
    };

    if backend_done {
        inner.on_success(&name);
    }

    let publish = backend_done && !stopped && !policy.altered() && !ephemeral && mode == BackendMode::Native;
    let mut record: Option<CheckpointRecord> = None;
    if publish {
        let key = match &promise {
            Some((key, _)) => key.clone(),
            None => next_hash(ctx.hashes.last().map(String::as_str), &policy.assistant_message()),
        };
        match control.freeze().await {
            Ok(snapshot) => {
                let now = now_ms();
                match inner.store.insert_checkpoint(&name, &key, &snapshot, now) {
                    Ok(id) => {
                        record = Some(CheckpointRecord {
                            id,
                            backend: name.clone(),
                            prefix_hash: key,
                            snapshot,
                            created_at: now,
                            last_used_at: now,
                        })
                    }
                    Err(e) => {
                        tracing::warn!(backend = %name, error = %e, "indexing checkpoint failed");
                        slot.backend.delete_snapshot(&snapshot).await;
                    }
                }
            }
            Err(e) => tracing::warn!(backend = %name, error = %e, "freeze failed; no checkpoint"),
        }
    } else {
        control.discard().await;
    }
    // The backend is done with this turn: free the slot first, so a request waiting on
    // the promise can take it as soon as it wakes (max_concurrency 1 must not stall).
    drop(permit.take());
    if let Some((_, guard)) = promise.take() {
        guard.resolve(record);
    }
    // `_guard` keeps the base marked in use until here, so GC never deletes it mid-turn.
    Attempt::Completed { finish, usage }
}
