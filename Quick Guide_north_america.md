# 🔥 Quick Start Guide - North America Fire Analysis v1.4.3

**5分で始める北米火災分析システム**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NASA FIRMS](https://img.shields.io/badge/Data-NASA%20FIRMS-red.svg)](https://firms.modaps.eosdis.nasa.gov/)

## 🚀 超速スタート

### ステップ1: 環境準備 (1分)
```bash
# 1. リポジトリをクローン
git clone https://github.com/yourusername/north-america-fire-analysis-v1-4-3.git
cd north-america-fire-analysis-v1-4-3

# 2. 自動セットアップ実行
python setup.py
```

### ステップ2: API設定 (1分)
```bash
# NASA FIRMS API キーを取得
# https://firms.modaps.eosdis.nasa.gov/api/

# 設定ファイルを編集
notepad config\config_north_america_firms.json
```

設定ファイル内の`map_key`を更新：
```json
{
  "nasa_firms": {
    "map_key": "YOUR_API_KEY_HERE"
  }
}
```

### ステップ3: 実行 (3分)
```bash
# メインパイプライン実行
python north_america_firms_pipeline_v143.py
```

## 📊 即座に得られる結果

### 🎯 分析ファイル
```
results/
├── nasa_firms_data.csv                      # 火災データ
├── tsne_plot.png                           # クラスター可視化
├── cluster_geographic_distribution.png      # 地理的分布
├── cluster_temporal_patterns.png           # 時間パターン
├── cluster_intensity_analysis.png          # 強度分析
└── comprehensive_fire_analysis_report.md   # 包括的レポート
```

### 📈 サンプル統計（標準設定）
- **処理データ**: 20,000+ 火災検知
- **クラスター数**: 15個程度
- **処理時間**: ~2分
- **品質スコア**: 0.6-0.8

## 🌎 対象地域

```
┌─────────────────────────────────────┐
│ 北米地域 (25°N-70°N, 170°W-50°W)    │
├─ Alaska            (極北・森林)      │
├─ Western Canada    (BC・Alberta)    │
├─ Central Canada    (Prairie州)      │
├─ Eastern Canada    (Quebec・大西洋州) │
├─ Western USA       (太平洋岸・山岳)   │
├─ Midwest USA       (五大湖・平原)    │
├─ Southern USA      (南部・湾岸)      │
├─ Eastern USA       (北東部・中部大西洋)│
└─ Hawaii            (太平洋諸島)      │
└─────────────────────────────────────┘
```

## ⚡ カスタマイズ

### 分析期間を変更
```json
{
  "nasa_firms": {
    "days_back": 5    // 5日間に短縮
  }
}
```

### 信頼度しきい値を調整
```json
{
  "nasa_firms": {
    "confidence_threshold": 80    // 高精度データのみ
  }
}
```

### 地域を絞り込み
```json
{
  "nasa_firms": {
    "area_params": {
      "north": 50,   // カナダ南部のみ
      "south": 25,
      "east": -60,   // 東海岸まで
      "west": -140   // 西海岸から
    }
  }
}
```

## 🎨 結果の見方

### 1. t-SNE プロット (`tsne_plot.png`)
- **色分け**: 異なるクラスター
- **近接性**: 類似した火災パターン
- **分離度**: クラスターの明確さ

### 2. 地理的分布 (`cluster_geographic_distribution.png`)
- **大きさ**: 火災件数
- **位置**: クラスター重心
- **色**: 地域分類

### 3. 時間パターン (`cluster_temporal_patterns.png`)
- **時間軸**: 24時間・7日間サイクル
- **ピーク**: 活動最大時間帯
- **周期性**: 規則的パターン

### 4. 強度分析 (`cluster_intensity_analysis.png`)
- **明度**: 火災温度 (K)
- **信頼度**: 検知精度 (%)
- **カテゴリ**: 高/中/低強度分類

## 🚨 トラブルシューティング

### よくある問題と解決法

#### ❌ API エラー
```
Error: Invalid API key
```
**解決**: config/config_north_america_firms.json の map_key を確認

#### ❌ メモリ不足
```
Error: Memory allocation failed
```
**解決**: processing.max_samples を減らす (例: 10000)

#### ❌ 依存関係エラー
```
ModuleNotFoundError: No module named 'faiss'
```
**解決**: `pip install -r requirements.txt` を再実行

#### ❌ データなし
```
Warning: No fire data found
```
**解決**: 期間を延長 (days_back を増加)

## 📱 使用例シナリオ

### 🔍 緊急事態対応
```bash
# 高信頼度・直近データで緊急分析
python -c "
import json
config = json.load(open('config/config_north_america_firms.json'))
config['nasa_firms']['days_back'] = 3
config['nasa_firms']['confidence_threshold'] = 80
json.dump(config, open('config/config_north_america_firms.json', 'w'), indent=2)
"
python north_america_firms_pipeline_v143.py
```

### 📊 週次レポート作成
```bash
# 週次分析設定
python -c "
import json
config = json.load(open('config/config_north_america_firms.json'))
config['nasa_firms']['days_back'] = 7
config['report']['region_name'] = 'North America Weekly'
json.dump(config, open('config/config_north_america_firms.json', 'w'), indent=2)
"
python north_america_firms_pipeline_v143.py
```

### 🌲 季節分析
```bash
# 長期トレンド分析
python -c "
import json
config = json.load(open('config/config_north_america_firms.json'))
config['nasa_firms']['days_back'] = 30
config['report']['region_name'] = 'North America Monthly'
json.dump(config, open('config/config_north_america_firms.json', 'w'), indent=2)
"
python north_america_firms_pipeline_v143.py
```

## 🎯 高度な機能

### バッチ処理
```bash
# 複数地域を順次分析
for region in "alaska" "western_usa" "eastern_canada"; do
    python scripts/region_specific_analysis.py --region $region
done
```

### 自動化スケジュール
```bash
# Windows タスクスケジューラ設定例
schtasks /create /tn "火災分析" /tr "python C:\path\to\north_america_firms_pipeline_v143.py" /sc daily /st 06:00
```

### 結果の統合
```bash
# 複数日の結果を統合分析
python scripts/trend_analysis.py --start-date 2024-01-01 --end-date 2024-01-31
```

## 📋 チェックリスト

### ✅ 実行前確認
- [ ] Python 3.8+ インストール済み
- [ ] NASA FIRMS API キー設定済み
- [ ] 必要パッケージインストール済み
- [ ] インターネット接続確認

### ✅ 実行後確認
- [ ] results/ ディレクトリにファイル生成
- [ ] ログファイルでエラーなし確認
- [ ] 可視化ファイルの妥当性確認
- [ ] レポート内容の検証

## 🔗 リンク集

### 📚 ドキュメント
- [完全マニュアル](README.md)
- [API仕様書](docs/api.md)
- [設定リファレンス](docs/config.md)

### 🌐 外部リソース
- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- [API登録](https://firms.modaps.eosdis.nasa.gov/api/)
- [データ仕様](https://firms.modaps.eosdis.nasa.gov/descriptions/)

### 🆘 サポート
- [GitHub Issues](https://github.com/yourusername/north-america-fire-analysis-v1-4-3/issues)
- [ディスカッション](https://github.com/yourusername/north-america-fire-analysis-v1-4-3/discussions)

## 💡 Tips & Tricks

### ⚡ パフォーマンス最適化
```json
{
  "processing": {
    "max_samples": 15000,    // データ量制限
    "batch_size": 50        // バッチサイズ削減
  },
  "embedding": {
    "device": "cuda"        // GPU使用（利用可能な場合）
  }
}
```

### 🎨 可視化カスタマイズ
```json
{
  "visualization": {
    "figsize": [20, 16],    // 図のサイズ拡大
    "dpi": 300,            // 高解像度
    "color_scheme": "viridis"  // カラーマップ変更
  }
}
```

### 📊 レポート詳細化
```json
{
  "report": {
    "include_raw_data": true,     // 生データ含める
    "detailed_analysis": true,    // 詳細分析追加
    "export_format": ["md", "pdf"]  // 複数フォーマット
  }
}
```

---

**🔥 Happy Fire Analysis!**  
*5分で始めて、プロレベルの火災分析を実現*

**Generated by North America Fire Analysis System v1.4.3**