"""
Enhanced Cluster Feature Analyzer for Fire Data Analysis
========================================================

高度なクラスタリング結果の特徴分析および可視化モジュール

機能:
- 地理的分析（重心、分散度、密度）
- 時間的パターン分析（時間別、曜日別）
- 強度分析（明度・信頼度統計）
- 地域別分析（地域分類・多様性）
- 包括的可視化生成

作成者: Fire Analysis System
バージョン: 1.4.3
対象地域: 北米（USA・カナダ）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import logging
from datetime import datetime, timedelta

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterFeatureAnalyzer:
    """クラスター特徴分析クラス"""
    
    def __init__(self):
        """初期化"""
        self.analysis_results = {}
        logger.info("ClusterFeatureAnalyzer initialized")
    
    def analyze_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        クラスター総合分析実行
        
        Args:
            df: クラスタリング結果を含むDataFrame
            
        Returns:
            分析結果辞書
        """
        logger.info("Starting comprehensive cluster analysis...")
        
        # 各種分析実行
        self.analysis_results = {
            'geographic_analysis': self._analyze_geographic_features(df),
            'temporal_analysis': self._analyze_temporal_patterns(df),
            'intensity_analysis': self._analyze_intensity_features(df),
            'regional_analysis': self._analyze_regional_distribution(df),
            'cluster_summary': self._generate_cluster_summary(df)
        }
        
        logger.info("Cluster analysis completed successfully")
        return self.analysis_results
    
    def _analyze_geographic_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """地理的特徴分析"""
        logger.info("Analyzing geographic features...")
        
        clusters = df['cluster'].unique()
        clusters = clusters[clusters >= 0]  # ノイズ除外
        
        geographic_analysis = {}
        
        for cluster_id in clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            
            # 基本統計
            lat_mean = cluster_data['latitude'].mean()
            lon_mean = cluster_data['longitude'].mean()
            lat_std = cluster_data['latitude'].std()
            lon_std = cluster_data['longitude'].std()
            
            # 境界
            lat_min, lat_max = cluster_data['latitude'].min(), cluster_data['latitude'].max()
            lon_min, lon_max = cluster_data['longitude'].min(), cluster_data['longitude'].max()
            
            # 密度計算（点/平方度）
            area = (lat_max - lat_min) * (lon_max - lon_min)
            density = len(cluster_data) / max(area, 0.01)  # ゼロ除算防止
            
            # 主要地域分類
            primary_region = self._classify_region(lat_mean, lon_mean)
            
            geographic_analysis[f'cluster_{cluster_id}'] = {
                'centroid': {
                    'latitude': lat_mean,
                    'longitude': lon_mean
                },
                'spread': {
                    'lat_std': lat_std,
                    'lon_std': lon_std
                },
                'bounds': {
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lon_min': lon_min,
                    'lon_max': lon_max
                },
                'density': density,
                'size': len(cluster_data),
                'primary_region': primary_region
            }
        
        return geographic_analysis
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """時間的パターン分析"""
        logger.info("Analyzing temporal patterns...")
        
        clusters = df['cluster'].unique()
        clusters = clusters[clusters >= 0]
        
        temporal_analysis = {}
        
        # acq_dateカラムが存在するかチェック
        if 'acq_date' not in df.columns:
            logger.warning("acq_date column not found, creating from index...")
            # デフォルト日付を生成
            df = df.copy()
            df['acq_date'] = pd.date_range('2024-01-01', periods=len(df), freq='H')
        
        # acq_timeカラムが存在するかチェック
        if 'acq_time' not in df.columns:
            logger.warning("acq_time column not found, creating from index...")
            df = df.copy()
            df['acq_time'] = (np.arange(len(df)) % 24 * 100).astype(int)
        
        for cluster_id in clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            
            # 時間別分析
            if 'acq_time' in cluster_data.columns:
                # 時間を正規化（HHMMからHHに変換）
                hours = cluster_data['acq_time'].apply(lambda x: int(x / 100) if pd.notna(x) else 12)
                hourly_dist = hours.value_counts().to_dict()
                peak_hour = hours.mode().iloc[0] if len(hours.mode()) > 0 else None
                peak_count = hourly_dist.get(peak_hour, 0) if peak_hour is not None else 0
            else:
                hourly_dist = {}
                peak_hour = None
                peak_count = 0
            
            # 曜日別分析
            if 'acq_date' in cluster_data.columns:
                try:
                    dates = pd.to_datetime(cluster_data['acq_date'])
                    weekdays = dates.dt.dayofweek
                    daily_dist = weekdays.value_counts().to_dict()
                    start_date = dates.min().strftime('%Y-%m-%d')
                    end_date = dates.max().strftime('%Y-%m-%d')
                except:
                    daily_dist = {}
                    start_date = None
                    end_date = None
            else:
                daily_dist = {}
                start_date = None
                end_date = None
            
            temporal_analysis[f'cluster_{cluster_id}'] = {
                'hourly_distribution': hourly_dist,
                'daily_distribution': daily_dist,
                'peak_activity': {
                    'hour': peak_hour,
                    'count': peak_count
                },
                'time_span': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }
        
        return temporal_analysis
    
    def _analyze_intensity_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """強度特徴分析"""
        logger.info("Analyzing intensity features...")
        
        clusters = df['cluster'].unique()
        clusters = clusters[clusters >= 0]
        
        intensity_analysis = {}
        
        for cluster_id in clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            
            # 明度統計
            brightness_stats = {
                'mean': cluster_data['brightness'].mean(),
                'std': cluster_data['brightness'].std(),
                'min': cluster_data['brightness'].min(),
                'max': cluster_data['brightness'].max(),
                'median': cluster_data['brightness'].median()
            }
            
            # 信頼度統計
            confidence_stats = {
                'mean': cluster_data['confidence'].mean(),
                'std': cluster_data['confidence'].std(),
                'min': cluster_data['confidence'].min(),
                'max': cluster_data['confidence'].max(),
                'median': cluster_data['confidence'].median()
            }
            
            # 強度カテゴリ分類
            avg_brightness = brightness_stats['mean']
            avg_confidence = confidence_stats['mean']
            intensity_category = self._categorize_intensity(avg_brightness, avg_confidence)
            
            intensity_analysis[f'cluster_{cluster_id}'] = {
                'brightness': brightness_stats,
                'confidence': confidence_stats,
                'intensity_category': intensity_category,
                'high_confidence_ratio': len(cluster_data[cluster_data['confidence'] >= 80]) / len(cluster_data)
            }
        
        return intensity_analysis
    
    def _analyze_regional_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """地域分布分析"""
        logger.info("Analyzing regional distribution...")
        
        clusters = df['cluster'].unique()
        clusters = clusters[clusters >= 0]
        
        regional_analysis = {}
        
        for cluster_id in clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            
            # 各点の地域分類
            regions = []
            for _, row in cluster_data.iterrows():
                region = self._classify_region(row['latitude'], row['longitude'])
                regions.append(region)
            
            # 地域分布統計
            region_counts = Counter(regions)
            dominant_region = max(region_counts, key=region_counts.get)
            region_diversity = len(region_counts)
            cross_regional = region_diversity > 1
            
            regional_stats = {
                'regional_distribution': dict(region_counts),
                'dominant_region': dominant_region,
                'region_diversity': region_diversity,
                'cross_regional': cross_regional
            }
            
            regional_analysis[f'cluster_{cluster_id}'] = regional_stats
        
        return regional_analysis
    
    def _generate_cluster_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """クラスター要約生成"""
        logger.info("Generating cluster summary...")
        
        clusters = df['cluster'].unique()
        clusters = clusters[clusters >= 0]
        
        summary = {
            'total_clusters': len(clusters),
            'total_points': len(df[df['cluster'] >= 0]),
            'noise_points': len(df[df['cluster'] == -1]),
            'cluster_sizes': dict(df[df['cluster'] >= 0]['cluster'].value_counts()),
            'overview': {}
        }
        
        for cluster_id in clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            
            # 各クラスターの概要生成
            centroid_lat = cluster_data['latitude'].mean()
            centroid_lon = cluster_data['longitude'].mean()
            primary_region = self._classify_region(centroid_lat, centroid_lon)
            avg_brightness = cluster_data['brightness'].mean()
            avg_confidence = cluster_data['confidence'].mean()
            
            summary['overview'][f'cluster_{cluster_id}'] = {
                'description': f"Cluster {cluster_id}: {primary_region} region fires",
                'size': len(cluster_data),
                'centroid': f"({centroid_lat:.2f}°, {centroid_lon:.2f}°)",
                'avg_intensity': f"{avg_brightness:.1f}K brightness, {avg_confidence:.1f}% confidence",
                'characteristics': self._describe_cluster_characteristics(cluster_data)
            }
        
        return summary
    
    def _classify_region(self, lat: float, lon: float) -> str:
        """座標から地域分類"""
        # ヨーロッパ地域の詳細分類 (-25°W to 50°E, 34°N to 72°N)
        if -25 <= lon <= 50 and 34 <= lat <= 72:
            # 北欧 (Nordic Countries)
            if 60 <= lat <= 72:
                if 4 <= lon <= 32:  # ノルウェー、スウェーデン、フィンランド
                    return "Nordic (Scandinavia)"
                elif -25 <= lon <= 4:  # アイスランド、フェロー諸島
                    return "Nordic (North Atlantic)"
                else:
                    return "Nordic Region"
            
            # 西欧 (Western Europe)
            elif 48 <= lat <= 60 and -10 <= lon <= 15:
                if -10 <= lon <= 2 and 50 <= lat <= 60:  # イギリス・アイルランド
                    return "British Isles"
                elif 2 <= lon <= 8 and 48 <= lat <= 55:   # フランス・ベルギー・オランダ
                    return "Western Europe (Continental)"
                elif 5 <= lon <= 15 and 47 <= lat <= 55:  # ドイツ・スイス
                    return "Central Western Europe"
                else:
                    return "Western Europe"
            
            # 南欧 (Southern Europe)
            elif 34 <= lat <= 48:
                if -10 <= lon <= 4 and 34 <= lat <= 44:   # スペイン・ポルトガル
                    return "Iberian Peninsula"
                elif 2 <= lon <= 18 and 40 <= lat <= 48:  # イタリア・フランス南部
                    return "Mediterranean West"
                elif 18 <= lon <= 30 and 34 <= lat <= 48: # バルカン半島
                    return "Balkans"
                elif 19 <= lon <= 30 and 34 <= lat <= 42: # ギリシャ・南バルカン
                    return "Southeast Mediterranean"
                else:
                    return "Southern Europe"
            
            # 東欧 (Eastern Europe)
            elif 15 <= lon <= 50 and 45 <= lat <= 60:
                if 15 <= lon <= 25 and 48 <= lat <= 55:   # ポーランド・チェコ
                    return "Central Europe"
                elif 20 <= lon <= 35 and 44 <= lat <= 52: # ウクライナ・ベラルーシ
                    return "Eastern Europe (Central)"
                elif 35 <= lon <= 50 and 50 <= lat <= 60: # ロシア西部
                    return "Eastern Europe (Russia)"
                elif 20 <= lon <= 50 and 40 <= lat <= 50: # 黒海沿岸
                    return "Eastern Europe (Black Sea)"
                else:
                    return "Eastern Europe"
            
            # その他のヨーロッパ地域
            else:
                return "Europe (Other)"
        
        # 北米地域の詳細分類 (-170°W to -50°W, 25°N to 70°N)
        elif -170 <= lon <= -50 and 25 <= lat <= 70:
            # アラスカ
            if -170 <= lon <= -130 and 55 <= lat <= 70:
                return "Alaska"
            
            # カナダ
            elif -140 <= lon <= -50 and 45 <= lat <= 70:
                if -140 <= lon <= -110 and 50 <= lat <= 70:  # 西部カナダ
                    return "Western Canada"
                elif -110 <= lon <= -90 and 45 <= lat <= 70:  # 中央カナダ
                    return "Central Canada"
                elif -90 <= lon <= -50 and 45 <= lat <= 70:   # 東部カナダ
                    return "Eastern Canada"
                else:
                    return "Canada (Other)"
            
            # アメリカ本土
            elif -125 <= lon <= -65 and 25 <= lat <= 49:
                if -125 <= lon <= -100 and 32 <= lat <= 49:   # 西部USA
                    return "Western USA"
                elif -100 <= lon <= -90 and 25 <= lat <= 49:  # 中西部USA
                    return "Midwest USA"
                elif -90 <= lon <= -75 and 25 <= lat <= 49:   # 南部USA
                    return "Southern USA"
                elif -80 <= lon <= -65 and 25 <= lat <= 45:   # 東部USA
                    return "Eastern USA"
                else:
                    return "Continental USA"
            
            # ハワイ・その他の島嶼部
            elif -165 <= lon <= -154 and 18 <= lat <= 25:
                return "Hawaii"
            
            # その他の北米地域
            else:
                return "North America (Other)"
        
        # 南米地域の分類
        elif -82 <= lon <= -35 and -56 <= lat <= 15:
            if -30 <= lat <= 15 and -82 <= lon <= -50:
                if -70 <= lon <= -50 and -20 <= lat <= 10:
                    return "Brazil"
                elif -82 <= lon <= -70 and -25 <= lat <= 15:
                    return "Peru/Colombia/Ecuador"
                elif -70 <= lon <= -60 and -25 <= lat <= 5:
                    return "Bolivia/Paraguay"
                else:
                    return "Northern South America"
            elif -56 <= lat <= -20 and -82 <= lon <= -35:
                if -76 <= lon <= -66 and -56 <= lat <= -17:
                    return "Chile"
                elif -74 <= lon <= -53 and -56 <= lat <= -20:
                    return "Argentina"
                else:
                    return "Southern Cone"
            else:
                return "South America"
        
        # アジア太平洋地域の分類（既存）
        elif 70 <= lon <= 100 and 0 <= lat <= 40:
            return "India/Central Asia"
        elif 100 <= lon <= 140 and 20 <= lat <= 50:
            return "East Asia (China/Japan/Korea)"
        elif 95 <= lon <= 140 and -10 <= lat <= 20:
            return "Southeast Asia"
        elif 140 <= lon <= 180 and 30 <= lat <= 50:
            return "Northeast Asia/Russia"
        elif 110 <= lon <= 155 and -45 <= lat <= -10:
            return "Australia"
        elif 165 <= lon <= 180 and -50 <= lat <= -30:
            return "New Zealand"
        elif lat >= 45:
            return "Northern Asia/Siberia"
        elif lat <= -40:
            return "Southern Ocean Region"
        else:
            return "Other Region"
    
    def _categorize_intensity(self, brightness: float, confidence: float) -> str:
        """火災強度カテゴリ分類"""
        if brightness >= 350 and confidence >= 80:
            return "Very High Intensity"
        elif brightness >= 320 and confidence >= 70:
            return "High Intensity"
        elif brightness >= 310 and confidence >= 60:
            return "Medium Intensity"
        else:
            return "Low to Moderate Intensity"
    
    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame) -> str:
        """クラスター特徴記述生成"""
        size = len(cluster_data)
        avg_brightness = cluster_data['brightness'].mean()
        
        if size > 200:
            size_desc = "Large fire cluster"
        elif size > 50:
            size_desc = "Medium fire cluster"
        else:
            size_desc = "Small fire cluster"
        
        if avg_brightness > 330:
            intensity_desc = "with high-intensity fires"
        elif avg_brightness > 315:
            intensity_desc = "with moderate-intensity fires"
        else:
            intensity_desc = "with low-intensity fires"
        
        return f"{size_desc} {intensity_desc}"
    
    def create_feature_visualizations(self, analysis_results: Dict[str, Any], save_dir: str) -> List[str]:
        """特徴分析可視化作成"""
        import os
        
        logger.info("Creating feature analysis visualizations...")
        saved_files = []
        
        # 1. 地理的分布マップ
        geo_map_path = os.path.join(save_dir, "cluster_geographic_distribution.png")
        self._create_geographic_distribution_plot(analysis_results['geographic_analysis'], geo_map_path)
        saved_files.append(geo_map_path)
        
        # 2. 地域別クラスター分布
        regional_path = os.path.join(save_dir, "cluster_regional_analysis.png")
        self._create_regional_analysis_plot(analysis_results['regional_analysis'], regional_path)
        saved_files.append(regional_path)
        
        # 3. 強度分析チャート
        intensity_path = os.path.join(save_dir, "cluster_intensity_analysis.png")
        self._create_intensity_analysis_plot(analysis_results['intensity_analysis'], intensity_path)
        saved_files.append(intensity_path)
        
        # 4. 時間パターン分析
        temporal_path = os.path.join(save_dir, "cluster_temporal_patterns.png")
        self._create_temporal_patterns_plot(analysis_results['temporal_analysis'], temporal_path)
        saved_files.append(temporal_path)
        
        logger.info(f"Created {len(saved_files)} feature visualization files")
        return saved_files
    
    def _create_geographic_distribution_plot(self, geo_analysis: Dict, save_path: str):
        """地理的分布プロット作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        clusters = list(geo_analysis.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))
        
        # 1. クラスター重心位置
        for i, cluster in enumerate(clusters):
            stats = geo_analysis[cluster]
            ax1.scatter(stats['centroid']['longitude'], stats['centroid']['latitude'], 
                       s=stats['size']*2, c=[colors[i]], alpha=0.7, 
                       label=f"{cluster} ({stats['primary_region']})")
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Cluster Centroids and Sizes')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. クラスターサイズ比較
        sizes = [geo_analysis[cluster]['size'] for cluster in clusters]
        cluster_names = [cluster.replace('cluster_', 'C') for cluster in clusters]
        bars = ax2.bar(cluster_names, sizes, color=colors)
        ax2.set_title('Cluster Sizes')
        ax2.set_ylabel('Number of Fire Detections')
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01, 
                    str(size), ha='center', va='bottom')
        
        # 3. 地理的分散度
        lat_spreads = [geo_analysis[cluster]['spread']['lat_std'] for cluster in clusters]
        lon_spreads = [geo_analysis[cluster]['spread']['lon_std'] for cluster in clusters]
        ax3.scatter(lat_spreads, lon_spreads, s=sizes, c=colors, alpha=0.7)
        ax3.set_xlabel('Latitude Spread (std)')
        ax3.set_ylabel('Longitude Spread (std)')
        ax3.set_title('Cluster Geographic Spread')
        for i, cluster in enumerate(clusters):
            ax3.annotate(cluster.replace('cluster_', 'C'), 
                        (lat_spreads[i], lon_spreads[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # 4. 密度vs範囲
        densities = [geo_analysis[cluster]['density'] for cluster in clusters]
        lat_ranges = [geo_analysis[cluster]['bounds']['lat_max'] - geo_analysis[cluster]['bounds']['lat_min'] 
                     for cluster in clusters]
        ax4.scatter(lat_ranges, densities, s=sizes, c=colors, alpha=0.7)
        ax4.set_xlabel('Latitude Range (degrees)')
        ax4.set_ylabel('Fire Density')
        ax4.set_title('Cluster Density vs Geographic Range')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_regional_analysis_plot(self, regional_analysis: Dict, save_path: str):
        """地域分析プロット作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 地域別分布データの準備
        all_regions = set()
        for cluster_data in regional_analysis.values():
            all_regions.update(cluster_data['regional_distribution'].keys())
        
        clusters = list(regional_analysis.keys())
        region_matrix = np.zeros((len(clusters), len(all_regions)))
        
        for i, cluster in enumerate(clusters):
            for j, region in enumerate(all_regions):
                region_matrix[i, j] = regional_analysis[cluster]['regional_distribution'].get(region, 0)
        
        # 1. 地域別ヒートマップ
        sns.heatmap(region_matrix, 
                   xticklabels=list(all_regions), 
                   yticklabels=[c.replace('cluster_', 'C') for c in clusters],
                   annot=True, fmt='g', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Fire Distribution by Region and Cluster')
        ax1.set_xlabel('Regions')
        ax1.set_ylabel('Clusters')
        
        # 2. 支配的地域
        dominant_regions = [regional_analysis[cluster]['dominant_region'] for cluster in clusters]
        region_counts = Counter(dominant_regions)
        ax2.pie(region_counts.values(), labels=region_counts.keys(), autopct='%1.1f%%')
        ax2.set_title('Dominant Regions Across Clusters')
        
        # 3. 地域多様性
        diversity_scores = [regional_analysis[cluster]['region_diversity'] for cluster in clusters]
        cluster_names = [cluster.replace('cluster_', 'C') for cluster in clusters]
        bars = ax3.bar(cluster_names, diversity_scores)
        ax3.set_title('Regional Diversity by Cluster')
        ax3.set_ylabel('Number of Regions')
        for bar, score in zip(bars, diversity_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    str(score), ha='center', va='bottom')
        
        # 4. 跨地域クラスター
        cross_regional = [regional_analysis[cluster]['cross_regional'] for cluster in clusters]
        cross_counts = {'Single Region': cross_regional.count(False), 
                       'Multi-Region': cross_regional.count(True)}
        ax4.pie(cross_counts.values(), labels=cross_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Single vs Multi-Regional Clusters')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_intensity_analysis_plot(self, intensity_analysis: Dict, save_path: str):
        """強度分析プロット作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        clusters = list(intensity_analysis.keys())
        
        # データ準備
        brightness_means = [intensity_analysis[cluster]['brightness']['mean'] for cluster in clusters]
        confidence_means = [intensity_analysis[cluster]['confidence']['mean'] for cluster in clusters]
        cluster_names = [cluster.replace('cluster_', 'C') for cluster in clusters]
        
        # 1. 明度分布
        brightness_data = []
        cluster_labels = []
        for cluster in clusters:
            stats = intensity_analysis[cluster]['brightness']
            brightness_data.extend([stats['mean']] * 5)  # 簡易分布表現
            cluster_labels.extend([cluster.replace('cluster_', 'C')] * 5)
        
        sns.boxplot(x=cluster_labels, y=brightness_data, ax=ax1)
        ax1.set_title('Fire Brightness Distribution by Cluster')
        ax1.set_ylabel('Brightness (K)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 信頼度分布
        ax2.bar(cluster_names, confidence_means, alpha=0.7)
        ax2.set_title('Average Confidence by Cluster')
        ax2.set_ylabel('Confidence (%)')
        ax2.tick_params(axis='x', rotation=45)
        for i, conf in enumerate(confidence_means):
            ax2.text(i, conf + 1, f'{conf:.1f}%', ha='center', va='bottom')
        
        # 3. 明度vs信頼度散布図
        colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))
        for i, cluster in enumerate(clusters):
            ax3.scatter(brightness_means[i], confidence_means[i], 
                       s=100, c=[colors[i]], alpha=0.7, 
                       label=cluster.replace('cluster_', 'C'))
        ax3.set_xlabel('Average Brightness (K)')
        ax3.set_ylabel('Average Confidence (%)')
        ax3.set_title('Brightness vs Confidence by Cluster')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 強度カテゴリ分布
        intensity_categories = [intensity_analysis[cluster]['intensity_category'] for cluster in clusters]
        category_counts = Counter(intensity_categories)
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Fire Intensity Categories')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_patterns_plot(self, temporal_analysis: Dict, save_path: str):
        """時間パターンプロット作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        clusters = list(temporal_analysis.keys())
        
        # 1. 時間別活動パターン
        all_hours = range(24)
        for cluster in clusters:
            hourly_dist = temporal_analysis[cluster]['hourly_distribution']
            values = [hourly_dist.get(hour, 0) for hour in all_hours]
            ax1.plot(all_hours, values, marker='o', label=cluster.replace('cluster_', 'C'))
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Fire Detections')
        ax1.set_title('Hourly Fire Activity Patterns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 4))
        
        # 2. 曜日別活動パターン
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for cluster in clusters:
            daily_dist = temporal_analysis[cluster]['daily_distribution']
            values = [daily_dist.get(day, 0) for day in range(7)]
            ax2.plot(weekdays, values, marker='s', label=cluster.replace('cluster_', 'C'))
        
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Number of Fire Detections')
        ax2.set_title('Weekly Fire Activity Patterns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ピーク活動時間
        peak_hours = []
        cluster_names = []
        for cluster in clusters:
            peak_info = temporal_analysis[cluster]['peak_activity']
            if peak_info['hour'] is not None:
                peak_hours.append(peak_info['hour'])
                cluster_names.append(cluster.replace('cluster_', 'C'))
        
        if peak_hours:
            ax3.bar(cluster_names, peak_hours, alpha=0.7)
            ax3.set_title('Peak Activity Hours by Cluster')
            ax3.set_ylabel('Peak Hour')
            ax3.set_ylim(0, 23)
            for i, hour in enumerate(peak_hours):
                ax3.text(i, hour + 0.5, f'{hour:02d}:00', ha='center', va='bottom')
        
        # 4. 活動期間分析
        time_spans = []
        for cluster in clusters:
            span_info = temporal_analysis[cluster]['time_span']
            if span_info['start_date'] and span_info['end_date']:
                start = pd.to_datetime(span_info['start_date'])
                end = pd.to_datetime(span_info['end_date'])
                duration = (end - start).days
                time_spans.append(duration)
            else:
                time_spans.append(0)
        
        if any(span > 0 for span in time_spans):
            valid_clusters = [clusters[i].replace('cluster_', 'C') for i, span in enumerate(time_spans) if span > 0]
            valid_spans = [span for span in time_spans if span > 0]
            ax4.bar(valid_clusters, valid_spans, alpha=0.7)
            ax4.set_title('Fire Activity Duration by Cluster')
            ax4.set_ylabel('Duration (days)')
            for i, span in enumerate(valid_spans):
                ax4.text(i, span + max(valid_spans)*0.01, f'{span}d', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()