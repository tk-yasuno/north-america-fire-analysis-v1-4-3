# North America Fire Analysis v1.4.3

**Comprehensive Fire Detection Analysis System for USA & Canada**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NASA FIRMS](https://img.shields.io/badge/Data-NASA%20FIRMS-red.svg)](https://firms.modaps.eosdis.nasa.gov/)

## 🔥 Overview

北米地域（アメリカ合衆国・カナダ）における火災検知データの包括的クラスタリング分析システム。NASA FIRMS衛星データを活用し、機械学習による高度な分析を実施します。

### Key Features
- **Real-time Data**: NASA FIRMS VIIRS_SNPP_NRT satellite data
- **Geographic Coverage**: USA & Canada (25°N-70°N, 170°W-50°W)
- **Advanced ML**: FAISS k-means clustering with sentence-transformers
- **Comprehensive Analysis**: Geographic, temporal, intensity, and regional analysis
- **Professional Reports**: Automated Markdown report generation
- **Rich Visualizations**: t-SNE plots, geographic distribution, temporal patterns

## 🌍 Geographic Coverage

```
Coverage Area: North America
- Latitude: 25°N to 70°N
- Longitude: 170°W to 50°W

Regions Included:
├── Alaska
├── Western Canada (British Columbia, Alberta)
├── Central Canada (Saskatchewan, Manitoba, Ontario)
├── Eastern Canada (Quebec, Atlantic Provinces)
├── Western USA (California, Oregon, Washington, etc.)
├── Midwest USA (Great Plains, Great Lakes)
├── Southern USA (Texas, Florida, etc.)
├── Eastern USA (Northeast, Southeast)
└── Hawaii
```

## 📊 Analysis Capabilities

### 1. Geographic Analysis
- Cluster centroids and spatial distribution
- Geographic density analysis
- Regional fire pattern identification
- Multi-region cluster detection

### 2. Temporal Analysis
- Hourly fire activity patterns
- Daily/weekly trend analysis
- Peak activity time identification
- Fire duration analysis

### 3. Intensity Analysis
- Fire brightness (temperature) analysis
- Confidence level distribution
- Fire intensity categorization
- High-risk fire identification

### 4. Regional Characteristics
- 9 detailed North American regions
- Cross-regional fire pattern analysis
- Regional diversity scoring
- Dominant region identification

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/yourusername/north-america-fire-analysis-v1-4-3.git
cd north-america-fire-analysis-v1-4-3
pip install -r requirements.txt
```

### Basic Usage
```bash
python north_america_firms_pipeline_v143.py
```

### Advanced Configuration
```python
# Edit config/config_north_america_firms.json
{
  "nasa_firms": {
    "days_back": 10,
    "confidence_threshold": 60,
    "area_params": {
      "north": 70,
      "south": 25,
      "east": -50,
      "west": -170
    }
  }
}
```

## 📁 Project Structure

```
north-america-fire-analysis-v1-4-3/
├── north_america_firms_pipeline_v143.py  # Main pipeline
├── config/
│   └── config_north_america_firms.json   # Configuration
├── scripts/
│   ├── data_collector.py                 # NASA FIRMS data collection
│   ├── model_loader.py                   # ML model loading
│   ├── embedding_generator.py            # Text embeddings
│   ├── clustering.py                     # Clustering algorithms
│   ├── visualization.py                  # Visualizations
│   ├── adaptive_clustering_selector.py   # Adaptive clustering
│   ├── hdbscan_clustering.py            # HDBSCAN implementation
│   ├── cluster_feature_analyzer.py       # Feature analysis
│   └── fire_analysis_report_generator.py # Report generation
├── docs/                                 # Documentation
├── results/                             # Analysis outputs
└── README.md                            # This file
```

## 🛠️ Core Components

### 1. Data Collection (`data_collector.py`)
- NASA FIRMS API integration
- Automatic data filtering and validation
- Geographic boundary enforcement
- Confidence threshold application

### 2. Machine Learning Pipeline
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Clustering**: Adaptive FAISS k-means + HDBSCAN
- **Quality Metrics**: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index

### 3. Analysis Engine (`cluster_feature_analyzer.py`)
- Geographic distribution analysis
- Temporal pattern extraction
- Fire intensity categorization
- Regional classification system

### 4. Visualization (`visualization.py`)
- t-SNE 2D cluster visualization
- Geographic distribution maps
- Temporal pattern charts
- Intensity distribution plots

### 5. Report Generation (`fire_analysis_report_generator.py`)
- Comprehensive Markdown reports
- Executive summary generation
- Statistical analysis tables
- Visualization integration

## 📈 Sample Output

### Analysis Statistics
```
Total Fire Detections: 20,000+
Clusters Identified: 15
Quality Score: 0.672
Noise Ratio: 0.0%
Processing Time: ~122 seconds
```

### Generated Files
```
results/
├── nasa_firms_data.csv
├── tsne_plot.png
├── score_distribution.png
├── cluster_geographic_distribution.png
├── cluster_regional_analysis.png
├── cluster_intensity_analysis.png
├── cluster_temporal_patterns.png
└── comprehensive_fire_analysis_report.md
```

## 🔧 Configuration Options

### NASA FIRMS Settings
```json
{
  "nasa_firms": {
    "api_url": "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
    "map_key": "your_api_key",
    "satellite": "VIIRS_SNPP_NRT",
    "days_back": 10,
    "confidence_threshold": 60
  }
}
```

### ML Model Settings
```json
{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu"
  },
  "clustering": {
    "random_state": 42,
    "min_cluster_size": 10
  }
}
```

## 🌎 Regional Classification System

The system identifies 9 distinct North American regions:

1. **Alaska**: Arctic and subarctic regions
2. **Western Canada**: British Columbia, Alberta  
3. **Central Canada**: Saskatchewan, Manitoba, Ontario
4. **Eastern Canada**: Quebec, Atlantic provinces
5. **Western USA**: Pacific Coast, Mountain West
6. **Midwest USA**: Great Plains, Great Lakes
7. **Southern USA**: Texas, Southeast, Gulf Coast
8. **Eastern USA**: Northeast, Mid-Atlantic
9. **Hawaii**: Pacific islands

## 📊 Analysis Methodology

### 1. Data Preprocessing
- Geographic filtering (North America bounds)
- Confidence threshold application (≥60%)
- Data validation and cleaning

### 2. Feature Engineering
- Text description generation from fire attributes
- 384-dimensional embedding generation
- Spatial and temporal feature extraction

### 3. Clustering Analysis
- Adaptive algorithm selection (k-means/HDBSCAN)
- Quality-based optimization
- Noise detection and filtering

### 4. Feature Analysis
- Geographic centroids and spread
- Temporal activity patterns
- Intensity distribution analysis
- Regional characteristic extraction

## 🎯 Use Cases

### 1. Emergency Response
- Real-time fire cluster identification
- High-intensity fire prioritization
- Geographic resource allocation

### 2. Research & Analysis
- Fire pattern trend analysis
- Climate change impact assessment
- Regional fire behavior studies

### 3. Policy & Planning
- Fire prevention strategy development
- Cross-border coordination (USA-Canada)
- Resource allocation optimization

### 4. Environmental Monitoring
- Ecosystem impact assessment
- Air quality correlation analysis
- Biodiversity conservation planning

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 8GB+ RAM recommended
- Internet connection for NASA FIRMS API

### Python Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
requests>=2.25.0
sentence-transformers>=2.0.0
faiss-cpu>=1.7.0
hdbscan>=0.8.0
plotly>=5.0.0
```

## 🚨 API Key Setup

1. Register at [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/)
2. Obtain your API key
3. Update `config/config_north_america_firms.json`:
```json
{
  "nasa_firms": {
    "map_key": "YOUR_API_KEY_HERE"
  }
}
```

## 📝 Output Interpretation

### Cluster Quality Score
- **0.7-1.0**: Excellent clustering
- **0.5-0.7**: Good clustering  
- **0.3-0.5**: Fair clustering
- **<0.3**: Poor clustering

### Fire Intensity Categories
- **Very High (350K+, 80%+ confidence)**: Emergency response required
- **High (320K+, 70%+ confidence)**: Close monitoring needed
- **Medium (310K+, 60%+ confidence)**: Standard monitoring
- **Low (<310K)**: Routine observation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA FIRMS**: Fire data provision
- **Sentence Transformers**: Text embedding technology
- **FAISS**: Efficient similarity search
- **scikit-learn**: Machine learning algorithms

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- https://www.linkedin.com/in/yasunotkt/

## 🔮 Future Enhancements

- [ ] Real-time processing pipeline
- [ ] Web dashboard interface
- [ ] Mobile app integration
- [ ] Weather data correlation
- [ ] Predictive modeling
- [ ] International expansion

---

**Generated by North America Fire Analysis System v1.4.3**  
*Advancing fire safety through data science and machine learning*
