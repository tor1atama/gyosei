# Better-EBPM Demo

Streamlit ベースの EBPM（証拠に基づく政策立案）サンドボックスです。質的/量的データを統合し、複数エージェントがチャット UI 上で政策仮説・施策案・KPI シナリオを提案/批評/最適化します。

## 主な機能
- **ステージガイドとコメント記録**: 「問題意識→仮説→施策→検証→意思決定」の各段階に合わせたガイド/ボタンと、政策担当者メモ・レビューノートがセッション中に保持されます。
- **RLHF 風マルチエージェント合議**: `SolutionSynthesizerAgent` が提案者と批評者の往復（擬似 RLHF）でロジックツリーを進化させ、枝分かれや利害対立を強制します。批評ログと reward が UI に表示され、透明性を担保します。
- **検索/ダミーデータ連携**: 政策/論文検索が 10 秒以内に結果を返せない場合は DummyDataAgent が即席データを生成し、後段処理が詰まらないようにしています。
- **KPI ε制約最適化**: KPI しきい値と予算制約を編集し、PuLP による ε 制約法でターゲット KPI ごとの最適配分を計算。結果はターゲット別タブ、配分テーブル、KPI 予測チャートで確認できます。
- **リスク・資源可視化**: 予算推定、Sankey/バブル図、リスク Register、関係者テンプレ作成などのモジュールを保持したまま拡張しています。

## セットアップ
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit>=1.37 pandas==2.2.2 numpy==1.26.4 plotly>=5.24 requests>=2.31 openai>=1.13.3 python-dotenv duckduckgo_search trafilatura graphviz pulp fitz bs4 frontend tools
```
PDF 系処理には `frontend`, `tools`, `fitz`（PyMuPDF）も必要です。OpenAI API キーを `.env` などで設定してください。

## 実行
```bash
cd Better-EBPM
streamlit run chat_ebpm_demo.py
```

## 追加メモ
- ε 制約最適化は `pulp` が未インストールの場合、自動的にヒントのみを表示します。
- ロジックツリーの RLHF メタデータ（reward/critic log）は `s["rlhf_meta"]` に格納され、UI の各戦略セクションで確認できます。
- ステークホルダー向けテンプレ、意思決定メモ、ステージ別ノートは `st.session_state.context` に保存されるため、チャットのリセット後も参照可能です。
