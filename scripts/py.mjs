// Run the project's Python (.venv, or AITV_PYTHON) with the given arguments, from the project root.
//   node scripts/py.mjs backend/run.py --port 8765
//   node scripts/py.mjs -m unittest discover -s backend/tests
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const venv = process.platform === 'win32' ? join(root, '.venv', 'Scripts', 'python.exe') : join(root, '.venv', 'bin', 'python');
const py = process.env.AITV_PYTHON || (existsSync(venv) ? venv : process.platform === 'win32' ? 'python' : 'python3');
const child = spawn(py, process.argv.slice(2), {
  cwd: root,
  stdio: 'inherit',
  env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
});
child.on('error', (e) => {
  console.error(`could not run ${py}: ${e.message}`);
  process.exit(1);
});
child.on('close', (code) => process.exit(code ?? 0));
