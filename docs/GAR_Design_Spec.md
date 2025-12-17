# Ghost Assimilation Relay (GAR)

---

## 1. 概要

このドキュメントは、ペルソナベースの LLM オーケストレーション環境である Ghost Assimilation Relay (GAR, 幽体同化継電器)　の設計をまとめたものである。

GAR は、ペルソナ同化・文脈制御・感情変調を統合した拡張LLMオーケストレータである。内部的には persona_layer, style_layer, retriever_layer を統合し、OpenAI 互換APIを介して外部UI（例: OpenWebUI）からアクセスできる。

#### 名称(案)
| 区分 | 名称 | 用途 |
|------|------|------|
| PyPI パッケージ | `gar-llm` | 公開・配布 |
| import 名 | `garllm` | Python 内部インポート |
| CLI コマンド | `garllm` | 実行・デバッグ |
| リポジトリ名 | `gar-llm` | GitHub / GitLab 用 |


---

## 2. 目的
大規模言語モデル(LLM)の応答にGARを経由させることにより、指示プロンプトよりも高度な人格設定、文脈のコントロールを実現することを目的とする。

---

## 3. 実行環境の構造

実行環境のディレクトリ構造を示す。案のみで現状実装のない部分は(案)と記載する。現状実装しているが、制作の過程で変更がある可能性がある部分は(仮)と記載する。
```
/home/<管理ユーザー>/
│
├── control/                      # ソフトモジュール制御用プログラム
│   ├── vllm/                     # vLLM 管理
│   │   ├── .venv/                # vLLM 制御スクリプトの用仮想環境(vLLM本体の実行用とは分ける)
│   │   ├── scripts/              # vLLM 管理スクリプト群
│   │   └── systemd_units/        # systemd サービス設定
│   │       ├── env/              # 各モデルの環境変数設定 (*.env)
│   │       └── vllm@.service     # vLLM 用テンプレートユニット
│   └── ollama/                   # ollama 管理 (案)
│       └─── scripts/             # ollama 管理スクリプト群 (案)
│
├── llm/
│   └── vllm-spark/               # 使用アーキテクチャごとに必要に応じてビルドしたLLMサーバープログラム
│
├── modules/
│   └── gar-llm/
│       ├── .venv/                      # GAR 制御用仮想環境
│       └── src/                        # Ghost Assimilation Relay 本体
│           ├── garllm/                        # Ghost Assimilation Relay 本体
│           │   ├── gateway/                   # 通信制御
│           │   │   ├── relay_server.py        # GARメインサーバ
│           │   │   ├── websocket_bridge.py    # UI連携用（リアルタイム更新）(案)
│           │   │   └── api_router.py          # 外部API統合（REST拡張用）(案)
│           │   │
│           │   ├── persona_layer/             # ペルソナ生成
│           │   │   └── persona_generator.py   # ペルソナ生成スクリプト
│           │   │
│           │   ├── style_layer/               # 文体・感情変調、文脈の制御
│           │   │   ├── context_controller.py  # 人格の感情・関係制御用スクリプト
│           │   │   ├── response_modulator.py  # 応答文に人格を反映させるスクリプト
│           │   │   └── style_modulator.py     # 文章に人格を反映させるスクリプト(陳腐化)
│           │   │
│           │   ├── retriever_layer/           # 知識補完層、データの収集
│           │   │   ├── retriever.py           # キーワードをもとに Duck Duck Go で情報を収集する
│           │   │   ├── semantic_condenser.py  # 収集した情報を要約するスクリプト
│           │   │   └── thought_profiler.py    # 要約した情報から、思想を抽出するスクリプト
│           │   │
│           │   ├── utils/                     # 他のスクリプトが共通して参照する機能
│           │   │   ├── env_utils.py           # 入出力ファイルのディレクトリ情報
│           │   │   ├── llm_client.py          # vLLMか、ollamaかバックエンドを切り替えて、OpenAI互換で接続する
│           │   │   └── logger.py              # ログ出力処理
│           │   └── __init__.py                # GAR初期化・メタ情報 (案)
│           │
│           └── pyproject.toml                 # Pythonパッケージ用メタデータ
│   
├── data/
│   ├── retrieved/                              # retriever_layerから取得したデータ
│   ├── semantic/                               # retrievedからcondenser, semanitc_condenserで抽出した意味データ
│   ├── thoughts/                               # semanticsからthought_profilerで生成抽出した思想データ
│   ├── personas/                               # ペルソナ(人格)関連データ
│   │   ├── persona_<persona name>.json         # ペルソナデータ本体
│   │   ├── expression_<persona name>.json      # 表現ライブラリ
│   │   └── state_<persona name>.json           # 感情軸、関係軸状態量
│   │
│   ├── memory/                  # 会話ログ・短期/長期記憶（context_controller 出力）(案)
│   │   ├── short_term/
│   │   └── long_term/
│   │
│   └── embeddings/              # chroma/vector データベースの永続層 (案)
│
├─── webui/
│   ├── docker-compose.yml     # OpenWebUI docker コンテナ生成用
│   ├── actions/               ← Sora2など外部API統合
│   └── chroma/                ← Embedding永続化
│
├── logs/                         # ログ出力
└── docs/                         # ドキュメント・設計資料(案)
```

---

## 3. 処理

- **レイヤー独立性**：各レイヤーは独立した仮想環境で運用。

### 3.1 実行環境
| python 仮想環境 | 概要 |
|------------------|------|
| **vllm/.venv** | vLLM ベースのLLM実行環境|
| **gar-llm/.venv** | GAR実行環境|

### 3.2 レイヤー
| レイヤー | 概要 |
|------------------|------|
| **context_layer** | 外部知識・史実情報の検索および要約。(仮) |
| **persona_layer** | 人格生成処理 |
| **style_layer** | 文体調整・ペルソナに基づく言語変調処理。 |

---

## 4. vLLM の操作

- **統一管理**：vLLM は systemd 経由で複数モデルを同時稼働可能。
- **CLI 一元化**：`vllmctl` により start/stop/enable/disable/check などを統一操作。
- vLLM サーバーの制御は `vllmctl` により下記のスクリプト群を切替え、統合実行する。bash 補完に対応するため、`.bashrc` に PATH 登録および補完設定を含める。
- モデルごとの起動パラメータは .envファイルにより管理。

| スクリプト名 | 機能概要 |
|---------------|-----------|
| **vllmctl** | モデル起動・停止・再起動・状態確認を一元管理するラッパ。 |
| **auto_vllm_config.sh** | モデル設定を自動生成（起動は行わない）。 |
| **start_vllm.sh / stop_vllm.sh** | モデル単位で起動・停止（未指定時は全停止）。 |
| **enable_vllm.sh / disable_vllm.sh** | systemd サービスの有効化・無効化。 |
| **check_vllm.sh** | 実行中モデルの状態を確認（--diag は非ブロッキング）。 |
| **hf_get_model.sh / rm_model.sh** | モデルの取得および削除。 |
| **show_vllm_env.sh** | 現行モデルの環境変数を表示。 |
| **bash_completion** | `.bashrc` に登録することでモデル名の補完を有効化。 |

---

## 5. LLM クライアント統合

`modules/utils/llm_client.py` は、vLLM・Ollama・OpenAI などの複数バックエンドに対応した共通アクセスモジュールです。クラス構造を取らず、関数ベースの軽量実装で統一されています。内部で利用可能なバックエンドを自動検出し、適切なAPIを選択して呼び出します。

### 主な特徴
- `request_llm()` 関数を通して全バックエンドにアクセス。
- `backend="auto"` により、稼働中のバックエンドを自動検出。
- vLLM の場合は systemd 環境を `env_utils.get_base_url()` から参照し、ポートを自動特定。
- Ollama はローカルホスト（11434番）を固定ポートで使用。
- OpenAI API は `OPENAI_API_KEY` を環境変数から取得。

### 使用例
```python
from modules.utils.llm_client import request_llm

response = request_llm(
    backend="auto",
    prompt="織田信長の統治理念を説明せよ。",
    temperature=0.8
)
print(response)
```

### 内部処理の流れ
1. `_detect_backend()` でバックエンド候補を順次検査（vLLM → Ollama → OpenAI）。
2. 使用可能なサーバーに接続成功した時点で選択確定。
3. バックエンドごとのAPI形式を統一化して返却（テキスト文字列）。

### 例外処理
- 最大3回のリトライを実施。
- JSONDecodeError・Timeout・接続失敗などはログ出力の上、`None` を返す。

### CLI デバッグモード
```bash
python3 -m modules.utils.llm_client --prompt "Hello" --backend auto --debug
```
出力例：
```
[llm_client] Backend detected: vLLM (http://localhost:8000)
[llm_client] Response: こんにちは、織田信長です。
```

---

## 6. 人格データと処理

### 6.1 人格データの生成

OpenWebUI上で GAR コマンドの入力または、ターミナルからの各pythonモジュールの実行により、ペルソナデータの元となるデータ抽出およびペルソナのデータへの合成が可能となっている。下表に、人格データ生成の二つの方式をまとめる。基本となるフローは数の通りである。

| 方法 | 処理内容 | 特徴 |
|--------|------|----------|
| **A. 事前生成型（Prebuilt Persona Mode）** | retriever ～ generator を OpenWebUI 起動前に実行し、Persona JSON を生成・保存。GARを介し、応答時に適用する。 | 高速、安定、モデル負荷が小さい。最初の運用モードとして採用。 |
| **B. 動的生成型（On-demand Persona Mode）** | OpenWebUI 入力文中のコマンドに基づいて、retriever ～ generator を実行。生成後に persona を反映する。 | 柔軟、動的生成可能だが待ち時間が発生する。 |


```
retriever (ddgs)
   ↓
cleaner
   ↓
condenser (SudachiPy)
   ↓
semantic_condenser (LLM)
   ↓
thought_profiler  (LLM)
   ↓
persona_assimilator  (LLM)

```

---

### 6.3 GAR コマンド

- OpenWebUI の入力文の中に対応するカッコ (), {}, [], <> などを書き、そのなかにGARコマンド書式を書くことで、GARの機能を呼び出し、応答を制御することができる。
- チャット欄にGARペルソナコマンド(gar.persona:<ペルソナ名>)を入力することで、事前に人格データが生成されていない場合はその場で人格データを生成、事前に生成されている場合はその perosna JSONをアシスタントに即時適用する。

**例:LLMに送られる履歴付きのChat history, SYSTEMのメッセージはチャット欄には表示されない。**
```
USER:       今の日本について (gar.persona:徳川家康)さんはどのように思いますか。
SYSTEM:     ASSISTANTはここから 徳川家康 の人格として応答します。
ASSISTANT:  そ、織田信長様のご意見とな。余は、そなたの見る目が確かであることは認める。
```
---

### 6.5 ペルソナの切替え
- セッション中に再びGARペルソナコマンドを入力することにより、別の人格に切り替えることが可能。これにより、模擬的に複数人での談話が可能となる。

```
USER:       今の日本について (gar.persona:徳川家康)さんはどのように思いますか。
SYSTEM:     ASSISTANTはここから 徳川家康 の人格として応答します。
ASSISTANT:  そ、織田信長様のご意見とな。余は、そなたの見る目が確かであることは認める。
USER:       では、今の日本どう思う？ 
ASSISTANT:  余は思う。今の日本の民が、皆、勤めを果たし、平和を重んじているならば、それほかに望むものはない。
USER:       そうですか。では、(gar.persona:織田信長)さんはどう思う？
SYSTEM:     ASSISTANTはここから 織田信長 の人格として応答します。
ASSISTANT:  ふむ、織田信長としてか。貴殿の問いは、それだけだろうか？
USER:       はい。
ASSISTANT:　我が関心は天下統一にあり。
```

---

## 7. 感情軸・関係軸（Emotion / Relation Axes）
GAR（Ghost Assimilation Relay）では、それぞれのペルソナに感情軸（emotion_axes）」と「関係軸（relations）」パラメータを保持する。セッションをまたいでもこれらは維持・管理され、話法の設定に使用する。これにより、
* ペルソナの振る舞い（発話スタイル）を、状況と継続的な状態（対話の経時的な関係性・感情）に基づいて安定かつ再現性高く生成する
* LLM のスタイル変換モジュールへ与える制御信号を明確化し、複数ペルソナやペルソナ間の相互作用を扱いやすくする

具体的には、人格データごとにユーザーまた他のペルソナとの関係性を6つのパラメータで、ペルソナ本人の感情の8つのパラメータで state_<ペルソナ名>.json に保持しており、ユーザーとの応答や応答履歴中のペルソナの発話内容を用いて、context_controller により変化させる。ペルソナの発話は、これらのパラメータをresponse_modulatorにより反映させ、応答文を生成する。

---

### 用語定義

* emotion_axes: 単一ペルソナにおける**現在の内面的な感情傾向**を数値化した辞書。八つの数値（例: joy, trust, fear, surprise, sadness, disgust, anger, anticipation）に対して値を持つベクトルデータとして表現される。
* relations: ペルソナと他者（ユーザや他ペルソナ）間の関係性を表す辞書。キーは相手名、値は軸名→数値の辞書
* state file: 1ペルソナごとの JSON（state_ペルソナ.json）で、emotion_axes と relations を含む

---

### パラメータ
人格ごとに保持する state jsonでは、自己の内面的感情を表すemotion_axesパラメータと対話中のユーザーや他のペルソナとの外面的関係性を表すrelation_axesパラメータを持ち、これらはセッションをまたいで値を管理する。管理はcontext_controllerにより行う。

### 読み出し・更新
役割分離により競合を避け、state 管理を一箇所に集約する
* relay_server: ペルソナ切替完了後に state を読み込む。
* context_controller: ユーザ発話に応じて state を更新し、ファイルへ書き戻す。更新は常に読み込み→マージ→保存で行う

### 7.1 感情軸 (内面的パラメータ)

以下は、GAR（Ghost Assimilation Relay）における「感情軸（emotion_axes）」設計仕様の詳細です。本設計はプルチックの感情の円環モデル（Plutchik's Wheel of Emotions）を基礎としつつ、より多次元的で動的な感情表現を可能にするために再構築されています。

---

#### 7.1.1 背景と目的

従来のプルチックモデルでは、感情の強弱や対立関係を二次元的に表現することはできても、複雑な心理状態（例:「恐れと信頼が同居する」など）を精密に扱うことが困難でした。GAR の感情軸設計では、**8つの基本感情を独立した座標軸とみなし、8次元ベクトル空間として表現**することで、より豊かな内面表現を実現します。

---

#### 7.1.2. 基本モデル構造

感情は以下の8軸によって表現されます：

| 軸名                   | 説明                  | 最小値（-1.0）                           | 最大値（+1.0）                        |
| -------------------- | ------------------- | ----------------------------------- | -------------------------------- |
| **joy（喜び）**          | 快楽・幸福・満足感を示す。       | 深い喪失や絶望、喜びを感じる力を失った状態（例: 無感動、鬱的状態）。 | 恍惚、歓喜、幸福感があふれ出す状態。強烈な達成感や快感の爆発。  |
| **trust（信頼）**        | 愛着、確信      | 疑念、不信          | 深い信頼と愛着、絶対的な確信。精神的な一体感を感じるレベル。  |
| **fear（恐れ）**         | 不安、危険への警戒、本能的防衛反応。  | 無謀、無防備、恐れを完全に欠いた状態。                 | 強烈な恐怖、逃避衝動、理性を圧倒するパニック状態。        |
| **surprise（驚き）**     | 新奇性への反応、変化への感受性。    | 完全な無関心、刺激に対して反応が起きない状態。             | 圧倒的な驚愕や衝撃、現実感が揺らぐほどの混乱や興奮。       |
| **sadness（悲しみ）**     | 喪失、痛み、内省的沈静。        | 無感情、鈍麻、哀しみすら感じない虚無。                 | 深い絶望、涙が止まらないほどの悲嘆、心が崩壊しかけるほどの苦痛。 |
| **disgust（嫌悪）**      | 不快、拒絶、道徳的・感覚的な排他反応。 | 無関心、受容、他者を許容できる状態。                  | 激しい嫌悪感、拒絶反応、吐き気を催すほどの不快。         |
| **anger（怒り）**        | 攻撃性、苛立ち、自己防衛的反応。    | 完全な冷静、怒りを手放した受容状態。                  | 激昂、憤怒、制御不能な怒り。暴発寸前の感情エネルギー。      |
| **anticipation（期待）** | 未来への予感、緊張、希望、焦燥。    | 無関心、停滞、未来への諦観。                      | 興奮、昂揚、待ちきれないほどの期待。確信を伴う行動的熱意。    |

各軸は -1.0〜1.0 の範囲を取り、0 は中立を示します。正の値はその感情の高まり、負の値はその感情の欠如または反転方向（抑制・無関心・冷却など）を表します。


---

#### 7.1.3. 感情テンプレート（生成指針）

各軸は LLM の発話スタイル制御に利用され、テンプレート文によりプロンプト内で明示的に表現されます。強度は `weak` / `medium` / `strong` の三段階に分かれ、文体・語気・テンポを制御します。

```json
EMOTION_TEMPLATES = {
    "joy": {
        "weak": "穏やかで心が安らいでいるように話す。",
        "medium": "明るく軽やかに、自然と声に弾みが出るように話す。",
        "strong": "感情が高ぶり、嬉しさが抑えきれないように話す。"
    },
    "trust": {
        "weak": "落ち着きと安らぎを感じ、静かに穏やかに話す。",
        "medium": "安心と安定を感じながら、自然体でゆったりと話す。",
        "strong": "深い安心と充足感に包まれ、温かく穏やかに話す。"
    },
    "fear": {
        "weak": "慎重で緊張を感じながら、少し抑えた声で話す。",
        "medium": "不安と恐れが混ざり、言葉に張り詰めた緊張がにじむように話す。",
        "strong": "恐怖や焦りが支配し、呼吸が浅く断片的な口調で話す。"
    },
    "surprise": {
        "weak": "小さな驚きと興味を感じて、軽く反応するように話す。",
        "medium": "はっきりと驚きが現れ、テンポが速くなるように話す。",
        "strong": "強い衝撃や驚愕を受け、思わず声や語気が大きくなるように話す。"
    },
    "sadness": {
        "weak": "静かに沈み込み、少し間を置きながら話す。",
        "medium": "切なさや哀しみが声に滲み、ゆっくりとした調子で話す。",
        "strong": "深い悲嘆に包まれ、途切れ途切れにかすれるように話す。"
    },
    "disgust": {
        "weak": "軽い不快感を覚え、やや無関心な調子で話す。",
        "medium": "明確な嫌悪や拒否の感情があり、語気が鋭くなる。",
        "strong": "強烈な不快感や拒絶の感情が溢れ、言葉に荒さが出る。"
    },
    "anger": {
        "weak": "いら立ちを抑えつつ、声の強さにわずかな緊張がこもる。",
        "medium": "明確な怒りが湧き上がり、短く強い言葉で話す。",
        "strong": "激しい怒りに突き動かされ、荒く激しい調子で話す。"
    },
    "anticipation": {
        "weak": "少し先を思い描きながら、期待と集中を感じて話す。",
        "medium": "高揚した期待感があり、語気が前のめりになるように話す。",
        "strong": "確信と興奮に満ち、勢いよく先を語るように話す。"
    }
}
```

---

#### 7.1.4. モデル運用上の設計意図

##### 多次元化

* 8つの感情軸は互いに独立して変化可能とし、組み合わせによって膨大な心理状態を表現できる。
* 例えば `joy=0.7, trust=0.5, fear=0.3` は「幸福だが不安も感じている」状態を表す。

##### 変化制御

* context_controller がユーザ発話や他ペルソナとのやり取りを解析し、各軸の増減を制御する。
* 変化は連続的（例: ±0.1〜0.3）で、発話内容に応じて動的に更新される。

##### 発話スタイル変換

* modulate_response にて emotion_axes を参照し、テンプレートに基づいた表現スタイルを LLM に指示。
* 各軸の強度は「感情ベクトル → 言語スタイル変換」の重みづけとして機能する。

---

#### 7.1.5. 拡張性

* 軸の追加や結合（例: 「羞恥」「誇り」などの派生感情）は容易に拡張可能。
* 感情変化の学習データ蓄積により、各ペルソナ固有の反応傾向を自動調整することも視野に入れている。

---

#### 7.1.6. 今後の課題

* 感情軸間の非線形依存（例: 強い怒りは信頼を減らすなど）の数理モデル化
* ペルソナごとの感情反応係数の自動最適化
* 感情変化の可視化ツール（時系列グラフ・ヒートマップ）の導入

---


### 7.2 関係軸 (外面的パラメータ)

| 軸名                       | 意味領域          | 発話傾向       | 対照軸（反転）   |
| ------------------------ | ------------- | ---------- | --------- |
| **Trust（信頼）**            | 認知的安心／裏切りの回復力 | 肯定的・長文・安心感 | 疑念・確認・論理的 |
| **Familiarity（親しみ）**     | 距離・慣れ／他人行儀さ   | 砕けた口調・軽口   | 丁寧・説明的    |
| **Hostility（敵意）**        | 攻撃／受容         | 批判・皮肉・断定   | 緩衝・柔和     |
| **Dominance（支配）**        | 主導／従属         | 命令・断言・強勢   | 依存・傾聴     |
| **Empathy（共感）**          | 情動理解／冷淡       | 感情語・同調     | 客観・事務的    |
| **Instrumentality（功利性）** | 目的・条件性／利他的動機  | 取引・効率・条件提示 | 無償・感情的    |


注意: 具体的な鍵名（軸名）はプロジェクトで統一すること。値のレンジは統一（推奨: -1.0〜1.0）する。

---

### 7.3 感情軸・関係軸のデータモデル
State jsonに下記の項目を記録管理する。

```json
{
  "emotion_axes": {
    "joy": 0.001309161754835755,
    "trust": 0.0006066882494120049,
    "fear": -0.0004477371355770665,
    "surprise": 0.00042732601307887604,
    "sadness": -0.00045174215056245566,
    "disgust": -0.0005902549352433692,
    "anger": -0.00043949728880923836,
    "anticipation": 0.0008941207173551671
  },
  "relations": {
    "user": {
      "Trust": 0.13940007959999995,
      "Familiarity": 0.06996871889999999,
      "Hostility": -0.07773516929999999,
      "Dominance": -0.03626454029999998,
      "Empathy": 0.09328407959999999,
      "Instrumentality": 0.10247928869999999
    },
    "default": {
      "Trust": -0.020999999999999998,
      "Familiarity": -0.020999999999999998,
      "Hostility": -0.020999999999999998,
      "Dominance": -0.020999999999999998,
      "Empathy": -0.020999999999999998,
      "Instrumentality": -0.020999999999999998
    },
    "織田信長": {
      "Trust": 0.0,
      "Familiarity": 0.0,
      "Hostility": 0.0,
      "Dominance": 0.0,
      "Empathy": 0.0,
      "Instrumentality": 0.0,
      "Domin、Empathy": 0.0
    }
  },
}
```

## 7.2 人格における「相」の重ね合わせによる話法の表現

# 🧩 Persona Phase Superposition Specification v1.0

**（GAR人格ファイル拡張仕様）**

## 1️⃣ 目的

人格の発話表現を「離散的な相」ではなく、
**複数の話法相（Phase）を重ね合わせた確率場として表現**する。

* 各Phaseは、話法・語尾・テンション・感情アンカーをもつ“基底ベクトル”。
* 発話時には、感情軸・関係軸の状態から各Phaseの重み（w_i）が算出される。
* 出力はそれらの混合（weighted prompt composition）として生成される。

---

## 2️⃣ 構造概要

結論から言うと、**相（phase）は固定ではない。**
人格ごとに定義される「内的構造（様式の分割）」であって、
どのキャラクターも同じ数・同じ意味を持つとは限らない。

---

### 🧩 現在の仕様的前提（personaファイル基準）

`persona_織田信長.json` のようなファイルの中には
`"phases": { ... }` というセクションがあって、
そこに各「相（phase）」が列挙されている。

例（仮）：

```json
"phases": {
  "覇王相": {
    "description": "支配と威圧を体現する相。",
    "style_bias": {"Dominance": 0.8, "Hostility": 0.4, "Empathy": -0.5}
  },
  "静観相": {
    "description": "冷静で戦略的な観察者としての相。",
    "style_bias": {"Trust": 0.3, "Empathy": 0.2, "Instrumentality": 0.1}
  },
  "狂気相": {
    "description": "激情と直感に支配される相。",
    "style_bias": {"Hostility": 0.7, "Anger": 0.6, "Fear": 0.3}
  }
}
```

このように、

* 相の**名前・数・内容は人格固有**
* 各相が Emotion / Relation の「方向づけ」を持つ
* つまり **人格の内部構成（心の相空間）** として定義されている

---

### 🧠 設計的に言うと

| 層            | 機能                | 汎用性        |
| ------------ | ----------------- | ---------- |
| **Emotion**  | 普遍的な心理ベクトル（全人格共有） | グローバル      |
| **Relation** | 対人行動ベクトル（全人格共有）   | グローバル      |
| **Phase**    | 内部様式の分布（人格固有）     | ローカル（人格依存） |

---

### 🧮 Phase数の扱い

* 各 persona JSON の `"phases"` に含まれるキー数（=相の数）が、その人格の「相空間の次元数」になる。
* GARのコア処理では、`phase_weights` を **動的配列** として扱う。
* よって、3でも4でも8でも動くようにすべき。

つまり、コード的には：

```python
# state_xxx.json の一部
"phase_weights": {
    "覇王相": 0.6,
    "静観相": 0.3,
    "狂気相": 0.1
}
```

---

### 🌐 実装指針（次のステップ）

1. **personaごとのphase定義**を読み取って
   `state` に `phase_weights` を生成（動的キー数）。

2. **Emotion/Relationの変化量**に応じて、
   それぞれの `phase` の重みを `style_bias` に基づいて更新。
   → “覇王相”は支配性↑や怒り↑で増加、など。

3. **揺らぎ（fluctuation）**を付与して、
   単調に遷移しないようにする（例：±0.02ノイズ＋正規化）。

4. **出力プロンプトに反映**

   * 現在最も重い相を主要相として明示。
   * 残りの相も副相として数値比率を渡す。

---

### ✅ 結論

> 相の数は人格ごとに異なる。
> GAR側は可変長で扱う。
> 各相は Emotion/Relation の「内部投影」を通じて重みが変化する。

---

## 🧩 データ構造仕様

### ① persona_xxx.json

```json
{
  "persona_name": "織田信長",
  "core_profile": { ... },
  "style": { ... },
  "expression_bank": { ... },

  "phases": {
    "戦略相": {
      "description": "冷静に戦況を見通し、指揮官として理性で語る。",
      "style_bias": { "Trust": 0.4, "Dominance": 0.6, "Hostility": -0.2 },
      "emotion_bias": { "joy": 0.2, "trust": 0.5, "fear": -0.2 },
      "tone_hint": "理知的で落ち着いた語り口。短文で断定的。"
    },
    "覇王相": {
      "description": "己の覇道を語る。強い信念と威圧を帯びる相。",
      "style_bias": { "Dominance": 0.9, "Trust": 0.3, "Hostility": 0.4 },
      "emotion_bias": { "anger": 0.6, "anticipation": 0.4 },
      "tone_hint": "声高で断定的。「〜である」調を多用。"
    },
    "省察相": {
      "description": "戦や政治の裏に潜む孤独や理想への葛藤を語る。",
      "style_bias": { "Empathy": 0.4, "Dominance": -0.3 },
      "emotion_bias": { "sadness": 0.5, "trust": 0.2 },
      "tone_hint": "静かで低い調子。短文に余韻を残す。"
    }
  },

  "phase_dynamics": {
    "alpha": 0.3,
    "beta": 0.2,
    "gamma": 0.05,
    "temperature": 0.4
  }
}
```


---

### ② state_xxx.json

（phase 関連を追加）

```json
{
  "relation_axes": {
    "Trust": 0.12,
    "Familiarity": -0.08,
    "Hostility": 0.05,
    "Dominance": 0.30,
    "Empathy": 0.18,
    "Instrumentality": -0.04
  },
  "emotion_axes": {
    "joy": 0.25,
    "trust": 0.32,
    "fear": 0.10,
    "surprise": 0.05,
    "sadness": 0.12,
    "disgust": 0.00,
    "anger": 0.08,
    "anticipation": 0.28
  },
  "phase_weights": {
    "戦略相": 0.4,
    "覇王相": 0.4,
    "省察相": 0.2
  },
  "dominant_phase": "覇王相"
}
```

---

### ✅ これでできること

* `update_phase_weights()` が `persona["phases"]` を読み込み、
  Emotion／Relation の変化から soft-argmax で分布を更新。
* `state` 側に `phase_weights` と `dominant_phase` が自動反映。
* 出力プロンプトで主相を自然言語的に提示可能。

--- 

## 6️⃣ 将来的な自動生成との互換性

`persona_generator.py` から自動抽出する。(仮)
思想・話法・感情傾向を解析して、初期phaseを自動提案できる。

---

## ✅ 結論

> **Persona Phase Superposition Specification v1.0** により、
> GARの人格JSONは「複数相の重ね合わせ」を内包する構造を正式に持つ。
> 感情・関係軸に応じて位相が滑らかに変化する人格表現を可能にする。

---
