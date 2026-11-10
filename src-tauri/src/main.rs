// AITrainView desktop shell (Tauri).
//
// The window shows the Three.js frontend bundled with the app; all heavy work
// (safetensors reading, inference, fine-tuning) runs in the Python backend,
// which this process starts on launch and stops on exit.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Manager, RunEvent};
use tauri_plugin_dialog::DialogExt;

#[derive(Default)]
struct Backend {
    child: Option<Child>,
    port: u16,
    reused: bool,          // an AITrainView backend was already running on the port
    error: Option<String>, // why it could not be started
    log: Option<PathBuf>,
    python: Option<PathBuf>,
    root: Option<PathBuf>,
}

struct BackendState(Mutex<Backend>);

/// Shown by the frontend when it cannot reach the backend.
#[derive(Serialize)]
struct BackendStatus {
    port: u16,
    reused: bool,
    running: bool,
    exit_code: Option<i32>,
    error: Option<String>,
    log: Option<String>,
    log_tail: Option<String>,
    python: Option<String>,
    root: Option<String>,
}

enum PortUse {
    Free,
    Ours,    // answers /api/health as AITrainView
    Foreign, // some other program
}

fn home() -> PathBuf {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// ~/.aitrainview: also holds the backend's thumbnail cache.
fn data_dir() -> PathBuf {
    home().join(".aitrainview")
}

fn probe(port: u16) -> PortUse {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(300)) else {
        return PortUse::Free;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(1500)));
    let _ = s.set_write_timeout(Some(Duration::from_millis(500)));
    let req = format!("GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if s.write_all(req.as_bytes()).is_err() {
        return PortUse::Foreign;
    }
    let mut buf = Vec::new();
    let _ = s.take(64 * 1024).read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf);
    if text.starts_with("HTTP/1.1 200") && text.contains("\"app\":\"aitrainview\"") {
        PortUse::Ours
    } else {
        PortUse::Foreign
    }
}

fn free_port() -> Option<u16> {
    TcpListener::bind("127.0.0.1:0").ok()?.local_addr().ok().map(|a| a.port())
}

/// The port to use, and whether an AITrainView backend already listens there.
fn choose_port() -> (u16, bool) {
    let wanted = std::env::var("AITV_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8765);
    match probe(wanted) {
        PortUse::Ours => (wanted, true),
        PortUse::Free => (wanted, false),
        // taken by another program: run our backend elsewhere instead of talking to it
        PortUse::Foreign => (free_port().unwrap_or(wanted), false),
    }
}

/// Folder that contains `backend/run.py`.
fn project_root(app: &AppHandle) -> PathBuf {
    let has_backend = |d: &Path| d.join("backend").join("run.py").exists();
    if let Some(p) = std::env::var_os("AITV_ROOT") {
        return PathBuf::from(p);
    }
    // the source checkout this binary was built from (development / this PC)
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")); // <checkout>/src-tauri
    let src = manifest.parent().unwrap_or(manifest).to_path_buf();
    if has_backend(&src) {
        return src;
    }
    // an installed copy: the backend sources are bundled as resources
    if let Ok(dir) = app.path().resource_dir() {
        if has_backend(&dir) {
            return dir;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors() {
            if has_backend(dir) {
                return dir.to_path_buf();
            }
        }
    }
    src
}

fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// AITV_PYTHON, then <root>/.venv, then ~/.aitrainview/venv, then python on PATH.
fn python(root: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os("AITV_PYTHON") {
        return PathBuf::from(p);
    }
    for venv in [root.join(".venv"), data_dir().join("venv")] {
        let py = venv_python(&venv);
        if py.exists() {
            return py;
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

fn start_backend(app: &AppHandle, b: &mut Backend) {
    let root = project_root(app);
    let py = python(&root);
    b.root = Some(root.clone());
    b.python = Some(py.clone());

    // stdout / stderr go to a log file (a release build has no console); keep the previous run's
    let logs = data_dir().join("logs");
    let log = logs.join("backend.log");
    let _ = fs::create_dir_all(&logs);
    if log.exists() {
        let _ = fs::rename(&log, logs.join("backend.prev.log"));
    }
    let (out, err) = match File::create(&log).and_then(|f| Ok((f.try_clone()?, f))) {
        Ok((a, b)) => (Stdio::from(a), Stdio::from(b)),
        Err(_) => (Stdio::null(), Stdio::null()),
    };
    b.log = Some(log);

    let mut cmd = Command::new(&py);
    cmd.arg(root.join("backend").join("run.py"))
        .arg("--port")
        .arg(b.port.to_string())
        // the backend exits by itself if this process dies without killing it
        .arg("--parent-pid")
        .arg(std::process::id().to_string())
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    if !root.join("package.json").exists() {
        // installed copy: save trained adapters / models under Documents, not next to the program
        let docs = app.path().document_dir().unwrap_or_else(|_| home());
        cmd.env("AITV_OUTPUTS", docs.join("AITrainView"));
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    match cmd.spawn() {
        Ok(child) => b.child = Some(child),
        Err(e) => b.error = Some(format!("Python を起動できませんでした（{}）: {e}", py.display())),
    }
}

/// Last ~4 KB of a file, starting at a line boundary.
fn tail(path: &Path) -> Option<String> {
    let mut f = File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    f.seek(SeekFrom::Start(len.saturating_sub(4096))).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    Some(if len > 4096 { text.split_once('\n').map(|(_, t)| t.to_string()).unwrap_or(text) } else { text })
}

#[tauri::command]
fn backend_status(state: tauri::State<BackendState>) -> BackendStatus {
    let mut b = state.0.lock().unwrap();
    let (running, exit_code) = match b.child.as_mut().map(|c| c.try_wait()) {
        Some(Ok(None)) => (true, None),
        Some(Ok(Some(s))) => (false, s.code()),
        Some(Err(_)) => (false, None),
        None => (b.reused, None),
    };
    BackendStatus {
        port: b.port,
        reused: b.reused,
        running,
        exit_code,
        error: b.error.clone(),
        log_tail: b.log.as_deref().and_then(tail),
        log: b.log.as_ref().map(|p| p.display().to_string()),
        python: b.python.as_ref().map(|p| p.display().to_string()),
        root: b.root.as_ref().map(|p| p.display().to_string()),
    }
}

/// Start the backend again (after fixing the Python environment, or if it crashed).
#[tauri::command]
fn restart_backend(app: AppHandle, state: tauri::State<BackendState>) {
    let mut b = state.0.lock().unwrap();
    if let Some(mut child) = b.child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    b.error = None;
    b.reused = false;
    start_backend(&app, &mut b);
}

/// Native folder picker (model folder to open, or where to save training results).
#[tauri::command]
async fn pick_folder(app: AppHandle, title: Option<String>) -> Option<String> {
    app.dialog()
        .file()
        .set_title(title.unwrap_or_else(|| "モデルのフォルダを選択（config.json と *.safetensors があるフォルダ）".into()))
        .blocking_pick_folder()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.display().to_string())
}

fn main() {
    let (port, reused) = choose_port();
    // tell the bundled frontend where the backend is; see frontend/src/api.ts
    let env_script = format!("window.__AITV_PORT__ = {port};");
    tauri::Builder::default()
        // a second launch only brings the running window to the front (it would otherwise share
        // the first one's backend and lose it when the first window closes); must be registered first
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri::plugin::Builder::<tauri::Wry>::new("aitv-env").js_init_script(env_script).build())
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendState(Mutex::new(Backend { port, reused, ..Default::default() })))
        .invoke_handler(tauri::generate_handler![pick_folder, backend_status, restart_backend])
        .setup(move |app| {
            if !reused {
                let handle = app.handle().clone();
                let state = app.state::<BackendState>();
                start_backend(&handle, &mut state.0.lock().unwrap());
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = app.state::<BackendState>().0.lock().unwrap().child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}
