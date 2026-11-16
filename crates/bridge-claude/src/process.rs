//! Process-group helpers: claude runs in its own process group so that stopping it also
//! stops the parking MCP server it spawned.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::process::Child;

/// Shared handle to a running process group (0 = none / already reaped).
#[derive(Debug, Clone, Default)]
pub struct GroupHandle(Arc<AtomicI32>);

impl GroupHandle {
    pub fn set(&self, pgid: i32) {
        self.0.store(pgid, Ordering::SeqCst);
    }

    pub fn clear(&self) {
        self.0.store(0, Ordering::SeqCst);
    }

    pub fn get(&self) -> i32 {
        self.0.load(Ordering::SeqCst)
    }

    /// Sends `sig` to the whole group. No-op when nothing is running.
    pub fn signal(&self, sig: i32) {
        let pgid = self.get();
        if pgid > 0 {
            // SAFETY: plain syscall; a negative pid addresses the process group.
            unsafe {
                libc::kill(-pgid, sig);
            }
        }
    }
}

/// SIGTERM the group, wait up to `grace` for the leader, then SIGKILL the group and wait up
/// to `kill_wait` (`tuning.kill_wait_ms`) for it to be reaped.
/// Always sweeps the group with SIGKILL at the end so no MCP child survives.
pub async fn terminate(
    child: &mut Child,
    group: &GroupHandle,
    grace: Duration,
    kill_wait: Duration,
) -> Option<std::process::ExitStatus> {
    group.signal(libc::SIGTERM);
    let status = match tokio::time::timeout(grace, child.wait()).await {
        Ok(Ok(s)) => Some(s),
        _ => {
            group.signal(libc::SIGKILL);
            tokio::time::timeout(kill_wait, child.wait()).await.ok().and_then(Result::ok)
        }
    };
    group.signal(libc::SIGKILL);
    group.clear();
    status
}

/// Wait for a natural exit for up to `grace`, then terminate (SIGTERM grace and SIGKILL
/// wait both `kill_wait`).
pub async fn wait_or_terminate(
    child: &mut Child,
    group: &GroupHandle,
    grace: Duration,
    kill_wait: Duration,
) -> Option<std::process::ExitStatus> {
    match tokio::time::timeout(grace, child.wait()).await {
        Ok(Ok(s)) => {
            group.signal(libc::SIGKILL);
            group.clear();
            Some(s)
        }
        _ => terminate(child, group, kill_wait, kill_wait).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// A process group whose leader ignores SIGTERM (so only SIGKILL ends it).
    fn stubborn() -> (Child, GroupHandle) {
        let child = tokio::process::Command::new("sh")
            .args(["-c", "trap '' TERM; sleep 30"])
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let group = GroupHandle::default();
        group.set(child.id().unwrap() as i32);
        (child, group)
    }

    #[tokio::test]
    async fn terminate_uses_the_given_grace() {
        let (mut child, group) = stubborn();
        tokio::time::sleep(Duration::from_millis(100)).await; // let the trap install
        let t = Instant::now();
        let status = terminate(&mut child, &group, Duration::from_millis(200), Duration::from_secs(5)).await;
        let took = t.elapsed();
        assert!(status.is_some(), "reaped within kill_wait");
        assert!(took >= Duration::from_millis(200) && took < Duration::from_secs(3), "{took:?}");
        assert_eq!(group.get(), 0);
    }

    #[tokio::test]
    async fn wait_or_terminate_uses_grace_then_kill_wait() {
        let (mut child, group) = stubborn();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let t = Instant::now();
        let status = wait_or_terminate(&mut child, &group, Duration::from_millis(150), Duration::from_millis(150)).await;
        let took = t.elapsed();
        assert!(status.is_some());
        assert!(took >= Duration::from_millis(300) && took < Duration::from_secs(3), "{took:?}");
    }
}
