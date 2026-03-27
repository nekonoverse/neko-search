# neko-search

SentencePiece + BM25 全文検索マイクロサービス。[Nekonoverse](https://github.com/nekonoverse/nekonoverse) のノートを観測コーパスから学習した語彙で分割し、高速に検索・サジェストする。

## 機能

| 機能 | 説明 |
|------|------|
| SentencePiece Unigram LM | コーパスから語彙を自動学習 (デフォルト 8000 トークン) |
| BM25 転置インデックス | TF-IDF ベースのランキング検索 |
| Prefix サジェスト | document frequency ランキングによるトークン補完 |
| 非同期学習 | バックグラウンドスレッドで語彙再学習、ダウンタイムなし |
| 二世代切替 | SearchState アトミックスワップで検索を止めずにインデックス更新 |

## セットアップ

### Docker (推奨)

```bash
docker build -t neko-search .
docker run -v neko-search-data:/data -p 8002:8002 neko-search
```

### 直接実行

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8002
```

## API

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/health` | GET | ヘルスチェック (モデル状態、ドキュメント数、世代) |
| `/version` | GET | API バージョン |
| `/index` | POST | ノートをインデックスに追加 |
| `/bulk-index` | POST | ノートを一括追加 |
| `/index/{note_id}` | DELETE | ノートをインデックスから削除 |
| `/search` | GET | BM25 検索 |
| `/suggest` | GET | Prefix サジェスト |
| `/train` | POST | バックグラウンド語彙再学習を開始 |
| `/train/status` | GET | 学習状態を取得 |

### 検索

```bash
curl "http://localhost:8002/search?q=ねこ&limit=20"
```

```json
{ "note_ids": ["uuid1", "uuid2", ...], "total": 2 }
```

### サジェスト

```bash
curl "http://localhost:8002/suggest?q=ねこ&limit=10"
```

```json
{
  "suggestions": [
    { "token": "▁ねこ", "df": 42 },
    { "token": "▁ねこの", "df": 15 }
  ],
  "prefix": "▁ねこ"
}
```

### 学習

```bash
# 学習開始 (即座に返却)
curl -X POST "http://localhost:8002/train" -H "Content-Type: application/json" \
  -d '{"vocab_size": 8000}'

# 状態確認
curl "http://localhost:8002/train/status"
```

## 自動再学習

コーパスの成長に合わせて語彙を自動で再学習する。環境変数で制御:

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `AUTO_TRAIN_ENABLED` | `true` | 自動再学習の有効/無効 |
| `AUTO_TRAIN_INTERVAL` | `604800` (7日) | チェック間隔 (秒) |
| `AUTO_TRAIN_MIN_NEW_DOCS` | `1000` | 前回学習時からの最小新規文書数 |

両条件 (間隔経過 AND 新規文書数) を満たしたときにバックグラウンドで再学習が開始される。再学習中も検索は継続可能 (ゼロダウンタイム)。前世代のモデルは `sp.model.prev` として保持される。

## Nekonoverse との連携

メインサーバーの `.env` に追加:

```bash
NEKO_SEARCH_URL=http://neko-search:8002
```

未設定時やサービスダウン時は ILIKE フォールバック検索が使われる。

## テスト

```bash
python -m pytest test_main.py -v
```

## ライセンス

MIT
