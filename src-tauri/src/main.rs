// AITrainView desktop shell (Tauri).
//
// The window shows the Three.js frontend bundled with the app; all heavy work
// (safetensors reading, inference, fine-tuning) runs in the Python backend,
// which this process starts on launch and stops on exit.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent};
use tauri_plugin_dialog::DialogExt;

struct Backend(Mutex<Option<Child>>);

fn port() -> u16 {
    std::env::var("AITV_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8765)
}

/// Folder that contains `backend/run.py` (env override, the source tree, or next to the exe).
fn project_root() -> PathBuf {
    if let Ok(p) = std::env::var("AITV_ROOT") {
        return PathBuf::from(p);
    }
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    if src.join("backend").join("run.py").exists() {
        return src;
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors() {
            if dir.join("backend").join("run.py").exists() {
                return dir.to_path_buf();
            }
        }
    }
    src
}

fn python(root: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("AITV_PYTHON") {
        return PathBuf::from(p);
    }
    let venv = if cfg!(windows) {
        root.join(".venv").join("Scripts").join("python.exe")
    } else {
        root.join(".venv").join("bin").join("python")
    };
    if venv.exists() {
        venv
    } else {
        PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
    }
}

fn backend_alive(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn start_backend(port: u16) -> std::io::Result<Child> {
    let root = project_root();
    let mut cmd = Command::new(python(&root));
    cmd.arg(root.join("backend").join("run.py"))
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.spawn()
}

/// Native folder picker for choosing a local model directory.
#[tauri::command]
async fn pick_folder(app: tauri::AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .set_title("モデルのフォルダを選択（config.json と *.safetensors があるフォルダ）")
        .blocking_pick_folder()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.display().to_string())
}

fn main() {
    // Tell the bundled frontend which port the backend is on (AITV_PORT); see frontend/src/api.ts.
    let env_script = format!("window.__AITV_PORT__ = {};", port());
    tauri::Builder::default()
        .plugin(tauri::plugin::Builder::<tauri::Wry>::new("aitv-env").js_init_script(env_script).build())
        .plugin(tauri_plugin_dialog::init())
        .manage(Backend(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![pick_folder])
        .setup(|app| {
            let port = port();
            // reuse a backend that is already running (e.g. started by hand)
            if !backend_alive(port) {
                match start_backend(port) {
                    Ok(child) => *app.state::<Backend>().0.lock().unwrap() = Some(child),
                    Err(e) => eprintln!("failed to start the Python backend: {e}"),
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = app.state::<Backend>().0.lock().unwrap().take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}
