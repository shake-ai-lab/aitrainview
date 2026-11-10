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
- GPU 推奨（GTX 1660 SUPER 6GB で Qwen3-0.6B の推論・LoRA 学習を確認済み）。重みの 3D 表示だけなら torch 不要

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

Rust のビルド成果物の置き場所は `.cargo/config.toml` の `target-dir` で変えられます（この PC では C: の空きが少ないため `D:/aitrainview-build/target` にしています）。

## 起動

```bash
npm run dev            # 開発モード（Vite のホットリロード + Tauri のウィンドウ）
npm run app:build      # リリースビルド → <target-dir>/release/aitrainview.exe
npm run app:installer  # インストーラ（NSIS）も作る
```

アプリ（Rust / Tauri）は起動時に Python バックエンド（`.venv` の python で `backend/run.py`）を立ち上げ、終了時に止めます。
すでに 8765 番でバックエンドが動いていればそれを使います。

ブラウザで開発する場合は `npm run backend`（API: http://127.0.0.1:8765）と `npm run web`（http://localhost:5173）を別々に起動します。

## 使い方

1. 起動画面でモデルを指定: HF のID（例 `Qwen/Qwen3-0.6B`）なら自動ダウンロード、またはローカルのフォルダ（`config.json` と `*.safetensors` がある場所）
2. 3D ビュー: ドラッグで回転、右ドラッグで移動、ホイールでズーム、クリックで詳細、ダブルクリックでズーム、`↑↓` で層移動、`F` で全体
3. **推論タブ** →「GPUに読み込む」→ プロンプトを入れて「生成」
4. **学習タブ** → データセット（JSONL: `messages` / `instruction`+`output` / `prompt`+`completion` / `text`、またはプレーンテキスト）を入れて「学習開始」。学習後は推論タブで変化を確認でき、「保存」で PEFT 互換の LoRA を `outputs/` に書き出します

## 構成

```
backend/aitrainview/
  st_reader.py   safetensors を numpy で直接メモリマップ（bf16 / fp8 もデコード、torch 不要）
  arch.py        テンソル名 + config.json → 層・グループ・ヘッド・experts の構造
  thumbs.py      重み行列をブロック縮約してヒートマップ化（RMS / 平均 / 間引き / 最大|w|）
  runtime.py     torch で推論。全モジュールにフックして活性・Attention・logit lens を送信
  trainer.py     自前の軽量 LoRA / フル微調整。勾配ノルム・ΔW・サンプル生成を送信
  server.py      FastAPI（REST + WebSocket）。ビルド済みフロントも配信
frontend/src/
  scene/flowLayout.ts   フロー（計算グラフ）レイアウト
  scene/layout.ts       コンパクトレイアウト + 共通型
  scene/modelView.ts    InstancedMesh + 配列テクスチャで全パネルを描画、発光アニメーション
  ui/                   詳細・推論・学習パネル
src-tauri/src/main.rs   Tauri（Rust）: ウィンドウ、バックエンドの起動/終了、フォルダ選択ダイアログ
```

## 制限事項・メモ

- 対応形式は safetensors のみ（`.bin` / GGUF / GPTQ・AWQ などの量子化重みは未対応）
- Gemma 系は HF で利用規約への同意が必要なことがあります（起動画面の「HFトークン」に入力、または環境変数 `HF_TOKEN`）。bf16 非対応 GPU では fp16 だとオーバーフローしやすいため自動で fp32 にします
- 活性は 1 シーケンスあたり先頭 128 トークン、各ベクトル 256 チャネル（平均で縮約）まで表示します。MoE の experts の活性は表示しません（routing されたトークンのみを通るため）
- 重みのサムネイルは `~/.aitrainview/cache` にキャッシュされます
- Python の場所は環境変数 `AITV_PYTHON`、プロジェクトの場所は `AITV_ROOT`、ポートは `AITV_PORT` で変更できます
