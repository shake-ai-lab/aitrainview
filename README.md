# AITrainView

LLM の **重み・アーキテクチャ・推論・ファインチューニング** を 3D で見るデスクトップアプリ。
Hugging Face 形式（safetensors）のモデルを開くと、テンソル名と config.json からアーキテクチャを自動で組み立て、
全ての重み行列をヒートマップとして 3D 空間に配置します。Qwen / Gemma / Llama / Mistral / MoE / マルチモーダル（ビジョン塔）に対応しています（モデルごとの決め打ちはしていません）。

## 見えるもの

| 表示 | 内容 |
|---|---|
| **フロー（既定）** | 順伝播を上→下の計算グラフとして表示。右の柱が残差ストリーム。各層で `norm → x̂ → Wq/Wk/Wv（ヘッドごとに奥行き方向へ分割）→ Q/K/V → softmax(QKᵀ) → A·V → Wo → (+)`、続いて MLP `W_gate/W_up → gate/up → act·up → W_down → (+)` |
| **コンパクト** | 重みだけを層ごとのフロアに並べた全体像（大きなモデル向け） |
| 推論 | 活性ブロック（チャネル × トークン）に、生成中の実際の値が1トークンずつ書き込まれる。Attention の重み、各層の logit lens（その層の時点での次トークン予測）、次トークン候補 |
| 学習 | LoRA / フル微調整を実行し、勾配の大きさで重みが橙色に光る。ΔW（学習による変化）のヒートマップ、‖ΔW‖/‖W‖、loss 曲線、途中の生成サンプル |

## 必要環境

- Windows / macOS / Linux、Node.js 20+、Python 3.10+、Rust（stable）
- Windows は WebView2（Windows 11 は標準搭載）と Visual Studio の C++ ビルドツール
- GPU 推奨（GTX 1660 SUPER 6GB で Qwen3-0.6B の推論・LoRA 学習・フル微調整（一部の層）を確認済み）。重みの 3D 表示だけなら torch 不要

## セットアップ

```bash
# Python（プロジェクト直下に .venv を作る。アプリはここの python でバックエンドを起動します）
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r backend/requirements.txt

# フロントエンド / Tauri CLI
npm install
```

（macOS / Linux では `.venv/Scripts/python.exe` の代わりに `.venv/bin/python`）

Rust のビルド成果物の置き場所は `.cargo/config.toml` の `target-dir` で変えられます（この PC では C: の空きが少ないため `D:/aitrainview-build/target` にしています。このファイルは PC ごとの設定なので git には入れていません）。

## 起動

```bash
npm run dev            # 開発モード（Vite のホットリロード + Tauri のウィンドウ）
npm run app:build      # リリースビルド → <target-dir>/release/aitrainview.exe
npm run app:installer  # インストーラ（NSIS）も作る
npm run check          # 型チェック + バックエンドの単体テスト
```

アプリ（Rust / Tauri）は起動時に Python バックエンド（`backend/run.py`）を立ち上げ、終了時に止めます。

- 8765 番（`AITV_PORT`）ですでに AITrainView のバックエンドが動いていればそれを使います。別のプログラムがそのポートを使っている場合は、空いているポートで起動します
- アプリが強制終了・クラッシュしても、バックエンドは親プロセスの終了を検知して自分で終了します（GPU メモリを握ったまま残りません）
- バックエンドの出力は `~/.aitrainview/logs/backend.log`（前回分は `backend.prev.log`）に残ります。起動できなかったときは、画面に理由とログの末尾が出ます

ブラウザで開発する場合は `npm run backend`（API: http://127.0.0.1:8765）と `npm run web`（http://localhost:5173）を別々に起動します。

### インストーラで入れる場合

インストーラには、アプリ本体とバックエンドの Python ソース（`backend/`）が入ります。**Python と依存パッケージ（torch など数 GB）は入りません**。インストール先の PC で、次のどれかを用意してください。

1. `~/.aitrainview/venv` に venv を作り、`backend/requirements.txt`（と GPU を使うなら CUDA 版 torch）を入れる
2. すでにある Python 環境の python を、環境変数 `AITV_PYTHON` で指定する

学習結果の保存先の既定は、インストール版では `ドキュメント/AITrainView` です（ソースから動かす場合は `outputs/`）。

## 使い方

1. 起動画面でモデルを指定: HF のID（例 `Qwen/Qwen3-0.6B`）なら自動ダウンロード、またはローカルのフォルダ（`config.json` と `*.safetensors` がある場所）。規約同意が必要なモデル（Gemma など）は 🔑 に HF トークンを入れます
2. 3D ビュー: ドラッグで回転、右ドラッグで移動、ホイールでズーム、クリックで詳細、ダブルクリックでズーム、`↑↓` で層移動、`F` で全体
3. **推論タブ** →「GPUに読み込む」→ プロンプトを入れて「生成」
4. **学習タブ** → データセット（JSONL: `messages` / `instruction`+`output` / `prompt`+`completion` / `text`、またはプレーンテキスト）を入れて「学習開始」。学習後は推論タブで変化を確認できます
5. **保存**: LoRA は PEFT 互換のアダプタ（`adapter_model.safetensors` + `adapter_config.json`）、フル微調整は元のモデルと同じ形式のフォルダ一式（safetensors・config・トークナイザ）として書き出します。フル微調整の保存先は AITrainView でそのまま開けるので、変化した重みを 3D で確認できます
6. **リセット / 元に戻す**: LoRA は外すだけ、フル微調整はディスクから重みを読み込み直します。未保存の学習結果が消える操作（解放・再読み込み・別のモデルを開く）の前には確認が出ます

## 構成

```
backend/aitrainview/
  st_reader.py   safetensors を numpy で直接メモリマップ（bf16 / fp8 もデコード、torch 不要）
  arch.py        テンソル名 + config.json → 層・グループ・ヘッド・experts の構造
  thumbs.py      重み行列をブロック縮約してヒートマップ化（RMS / 平均 / 間引き / 最大|w|）
  runtime.py     torch で推論。全モジュールにフックして活性・Attention・logit lens を送信
  trainer.py     自前の軽量 LoRA / フル微調整。勾配ノルム・ΔW・サンプル生成を送信、保存
  server.py      FastAPI（REST + WebSocket）。ビルド済みフロントも配信
backend/tests/   単体テスト（GPU・モデルのダウンロード不要）
frontend/src/
  scene/flowLayout.ts   フロー（計算グラフ）レイアウト
  scene/layout.ts       コンパクトレイアウト + 共通型
  scene/modelView.ts    InstancedMesh + 配列テクスチャで全パネルを描画、発光アニメーション
  scene/viewer.ts       カメラ・操作・描画（変化があったときだけ描画）
  ui/                   詳細・推論・学習パネル
src-tauri/src/main.rs   Tauri（Rust）: ウィンドウ、バックエンドの起動/監視/終了、フォルダ選択ダイアログ
```

## 制限事項・メモ

- 対応形式は safetensors のみ（`.bin` / GGUF / GPTQ・AWQ などの量子化重みは未対応）
- Gemma 系は HF で利用規約への同意が必要なことがあります（🔑 の「HFトークン」に入力、または環境変数 `HF_TOKEN`）。bf16 非対応 GPU では fp16 だとオーバーフローしやすいため自動で fp32 にします
- フル微調整は fp32 か bf16 で読み込んだときだけ使えます（fp16 は不安定なため）。LoRA とフル微調整は同時には使えません
- 活性は 1 シーケンスあたり先頭 128 トークン、各ベクトル 256 チャネル（平均で縮約）まで表示します。MoE の experts の活性は表示しません（routing されたトークンのみを通るため）
- 重みのサムネイルは `~/.aitrainview/cache` にキャッシュされ、合計 512 MB を超えると古いものから消します（`AITV_CACHE_MB` で変更）
- API は、このアプリ・バックエンド自身が配信するページ・Vite の開発サーバーからのアクセスだけを受け付けます（ブラウザで開いた他のサイトからは呼べません）

### 環境変数

| 変数 | 内容 |
|---|---|
| `AITV_PORT` | バックエンドのポート（既定 8765） |
| `AITV_PYTHON` | バックエンドを動かす python |
| `AITV_ROOT` | `backend/run.py` があるフォルダ |
| `AITV_CACHE` / `AITV_CACHE_MB` | サムネイルキャッシュの場所 / 上限（MB） |
| `AITV_OUTPUTS` | 学習結果の既定の保存先 |
| `AITV_ALLOW_ORIGINS` | API を呼べるオリジンの追加（カンマ区切り） |
| `HF_TOKEN` | Hugging Face のトークン |
