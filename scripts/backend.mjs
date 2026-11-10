// Run only the Python backend (for browser-based development: `npm run web`).
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const venv = process.platform === 'win32' ? join(root, '.venv', 'Scripts', 'python.exe') : join(root, '.venv', 'bin', 'python');
const py = process.env.AITV_PYTHON || (existsSync(venv) ? venv : 'python');
const child = spawn(py, [join(root, 'backend', 'run.py'), ...process.argv.slice(2)], {
  stdio: 'inherit',
  env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
});
child.on('close', (code) => process.exit(code ?? 0));
