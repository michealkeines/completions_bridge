//! completions-bridge: an OpenAI-compatible chat completions API backed by agent CLIs.
//!
//! Subcommands:
//!   serve         run the HTTP server (default)
//!   selftest      run every backend's self-test and print the result
//!   print-config  print the default configuration as YAML
//!   mcp-park      internal: parking MCP server spawned by the claude backend

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use bridge_types::{Backend, ChatService, Config};
use clap::{Parser, Subcommand};
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "completions-bridge", version, about = "OpenAI-compatible completions API backed by agent CLIs")]
struct Cli {
    /// Config file (YAML). Default: ~/.completions-bridge/config.yaml if it exists, else built-in defaults.
    #[arg(long, short, global = true, env = "BRIDGE_CONFIG")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP server.
    Serve {
        /// Override `server.listen`.
        #[arg(long)]
        listen: Option<String>,
    },
    /// Run backend self-tests and print the resulting modes.
    Selftest,
    /// Print the default configuration.
    PrintConfig,
    /// Internal: parking MCP server (stdio) used by the claude backend.
    #[command(hide = true)]
    McpPark {
        #[arg(long)]
        tools: PathBuf,
    },
}

fn main() -> Result<()> {
    // Everything the bridge (and the claude processes it starts) writes holds code,
    // conversations and account context: owner-only by default.
    // SAFETY: umask is a plain process-wide syscall with no memory effects.
    unsafe {
        libc::umask(0o077);
    }
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("BRIDGE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(async move {
        match cli.command.unwrap_or(Command::Serve { listen: None }) {
            Command::McpPark { tools } => bridge_claude::run_park_server(&tools).await,
            Command::PrintConfig => {
                print!("{}", yaml_serde::to_string(&Config::default())?);
                Ok(())
            }
            Command::Selftest => {
                let (config, workdir) = load_config(cli.config.as_deref())?;
                let backends = build_backends(&config, &workdir)?;
                for b in &backends {
                    let mode = b.self_test().await;
                    println!("{:<12} version={:<12} mode={:?}", b.name(), b.version(), mode);
                }
                Ok(())
            }
            Command::Serve { listen } => {
                let (mut config, workdir) = load_config(cli.config.as_deref())?;
                if let Some(l) = listen {
                    config.server.listen = l;
                }
                serve(config, workdir).await
            }
        }
    })
}

fn expand_home(p: &Path) -> PathBuf {
    if let Ok(rest) = p.strip_prefix("~") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    p.to_path_buf()
}

fn load_config(path: Option<&Path>) -> Result<(Config, PathBuf)> {
    let default_path = expand_home(Path::new("~/.completions-bridge/config.yaml"));
    let config = match path {
        Some(p) => {
            let p = expand_home(p);
            let text = std::fs::read_to_string(&p).with_context(|| format!("reading config {}", p.display()))?;
            yaml_serde::from_str(&text).with_context(|| format!("parsing config {}", p.display()))?
        }
        None if default_path.exists() => {
            let text = std::fs::read_to_string(&default_path)?;
            yaml_serde::from_str(&text).with_context(|| format!("parsing config {}", default_path.display()))?
        }
        None => Config::default(),
    };
    validate_config(&config)?;
    let workdir = expand_home(&config.server.workdir);
    std::fs::create_dir_all(&workdir).with_context(|| format!("creating workdir {}", workdir.display()))?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&workdir, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("restricting workdir {}", workdir.display()))?;
    }
    Ok((config, workdir))
}

fn validate_config(config: &Config) -> Result<()> {
    if config.models.is_empty() {
        bail!("config: no models defined");
    }
    for (id, route) in &config.models {
        if let Some(b) = &route.backend {
            if !config.backends.contains_key(b) {
                bail!("config: model {id} routes to unknown backend {b}");
            }
        }
    }
    if !config.backends.values().any(|b| b.enabled) {
        bail!("config: no enabled backends");
    }
    Ok(())
}

fn build_backends(config: &Config, workdir: &Path) -> Result<Vec<Arc<dyn Backend>>> {
    let exe = std::env::current_exe().context("locating own executable")?;
    let mut out: Vec<Arc<dyn Backend>> = Vec::new();
    for (name, cfg) in &config.backends {
        if !cfg.enabled {
            continue;
        }
        let backend: Arc<dyn Backend> = match cfg.kind.as_str() {
            "claude-cli" => Arc::new(bridge_claude::ClaudeBackend::new(
                name,
                cfg,
                workdir,
                vec![exe.clone(), PathBuf::from("mcp-park")],
            )?),
            "fake" => {
                let dir = workdir.join("fake").join(name);
                std::fs::create_dir_all(&dir)?;
                Arc::new(bridge_engine::fake::FakeBackend::new(name, cfg, &dir)?)
            }
            other => bail!("config: backend {name} has unknown kind {other:?}"),
        };
        out.push(backend);
    }
    Ok(out)
}

async fn serve(config: Config, workdir: PathBuf) -> Result<()> {
    let backends = build_backends(&config, &workdir)?;
    let state_dir = workdir.join("state");
    std::fs::create_dir_all(&state_dir)?;
    let engine = bridge_engine::Engine::new(&config, &state_dir, backends).await?;
    let app = bridge_openai::router(engine.clone(), &config);

    let listener = tokio::net::TcpListener::bind(&config.server.listen)
        .await
        .with_context(|| format!("binding {}", config.server.listen))?;
    let addr = listener.local_addr()?;
    if !addr.ip().is_loopback() {
        warn!("listening on non-loopback address {addr}; make sure server.token is set");
    }
    if config.server.token.is_none() {
        warn!("server.token is not set; any local process can use this API");
    }
    info!("completions-bridge listening on http://{addr}/v1");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            // SIGINT (Ctrl-C) and SIGTERM (kill, service managers) both shut down cleanly,
            // so running turns get to stop their claude processes.
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = async { match term.as_mut() { Some(t) => { t.recv().await; } None => std::future::pending::<()>().await } } => {}
            }
            info!("shutting down");
        })
        .await?;

    // Turns that already answered may still be freezing their checkpoint; give them a
    // moment (`server.shutdown_grace_s`) so a restart doesn't throw away warm state.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(config.server.shutdown_grace_s);
    loop {
        let in_flight: u64 = engine.status()["backends"]
            .as_array()
            .map(|bs| bs.iter().filter_map(|b| b["in_flight"].as_u64()).sum())
            .unwrap_or(0);
        if in_flight == 0 || std::time::Instant::now() >= deadline {
            if in_flight > 0 {
                warn!("exiting with {in_flight} turn(s) still in flight");
            }
            break;
        }
        // Poll interval: an implementation detail, not a tuning knob.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    Ok(())
}
