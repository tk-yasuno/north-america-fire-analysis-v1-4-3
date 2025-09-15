#!/usr/bin/env python3
"""
北米地域 NASA FIRMS 森林火災データ分析パイプライン v1.4.3
USA・カナダをカバーする北米エリアの火災データ分析システム
結果は自動生成されるタイムスタンプ付きフォルダに保存

エリア範囲:
- 緯度: 北緯25°～北緯70° (南部USA～北極圏カナダ)  
- 経度: 西経170°～西経50° (アラスカ～東海岸)
- 対象国: アメリカ合衆国、カナダ
"""

import os
import sys
import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from typing import Dict, List, Optional

# パス設定
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.append(scripts_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# モジュールインポート
from scripts.data_collector import DataCollector
from scripts.model_loader import ModelLoader
from scripts.embedding_generator import EmbeddingGenerator
from adaptive_clustering_selector import AdaptiveClusteringSelector
from scripts.visualization import VisualizationManager
from cluster_feature_analyzer import ClusterFeatureAnalyzer
from fire_analysis_report_generator import FireAnalysisReportGenerator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def _time_step(step_name):
    """ステップ実行時間測定用デコレーター"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            logger.info(f"=== Starting: {step_name} ===")
            try:
                result = func(self, *args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"✅ Completed: {step_name} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ Failed: {step_name} ({elapsed:.2f}s) - {str(e)}")
                raise
        return wrapper
    return decorator


class NorthAmericaFIRMSPipeline:
    """北米地域 FIRMS 火災分析パイプライン"""
    
    def __init__(self, config_file: str = "config/config_north_america_firms.json"):
        """
        Args:
            config_file: 設定ファイルパス
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.setup_output_directory()
        
        # パイプライン統計
        self.stats = {
            'start_time': datetime.now(),
            'total_samples': 0,
            'final_samples': 0,
            'processing_steps': [],
            'performance_metrics': {}
        }
        
    def _load_config(self) -> Dict:
        """設定ファイル読み込み"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded: {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def setup_output_directory(self):
        """出力ディレクトリ設定"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        base_dir = self.config['adaptive_clustering']['output_dir']
        self.output_dir = f"{base_dir}_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.output_dir}/cleaned", exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        
    @_time_step("Initializing Pipeline Components")
    def _initialize_components(self):
        """パイプラインコンポーネント初期化"""
        logger.info("=== Initializing Pipeline Components ===")
        
        # モデルローダー初期化
        self.model_loader = ModelLoader()
        
        # モデルロード
        model = self.model_loader.load_model()
        
        # 埋め込み生成器初期化  
        self.embedding_generator = EmbeddingGenerator(
            model=model,
            output_dir=self.output_dir
        )
        
        # 可視化マネージャー初期化
        self.visualization_manager = VisualizationManager(
            output_dir=self.output_dir
        )
        
        logger.info("All components initialized successfully")
    
    @_time_step("Collecting NASA FIRMS Data (North America Region)")
    def _collect_nasa_firms_data(self) -> Dict:
        """NASA FIRMSデータ収集 - 北米地域"""
        logger.info("=== Collecting NASA FIRMS Data (North America Region) ===")
        
        # データ収集器初期化
        collector = DataCollector()
        
        # 北米エリア設定
        area_params = self.config['nasa_firms']['area_params']
        print(f"Fetching NASA FIRMS data for past {self.config['nasa_firms']['days_back']} days")
        print(f"Area: {area_params}")
        
        # NASA FIRMS API URL構築
        api_url = (
            f"{self.config['nasa_firms']['data_source']}"
            f"{self.config['nasa_firms']['map_key']}/"
            f"{self.config['nasa_firms']['satellite']}/"
            f"{area_params['west']},{area_params['south']},"
            f"{area_params['east']},{area_params['north']}/"
            f"{self.config['nasa_firms']['days_back']}"
        )
        print(f"API URL: {api_url}")
        
        # データ収集実行
        nasa_data = collector.collect_nasa_firms_data(
            area_params=area_params,
            days_back=self.config['nasa_firms']['days_back'],
            map_key=self.config['nasa_firms']['map_key']
        )
        
        if nasa_data is None or len(nasa_data) == 0:
            raise ValueError("No NASA FIRMS data collected")
        
        print(f"Successfully collected {len(nasa_data)} fire detection records")
        
        # 信頼度フィルタリング
        confidence_threshold = self.config['nasa_firms']['confidence_threshold']
        filtered_data = nasa_data[nasa_data['confidence'] >= confidence_threshold]
        
        print(f"Filtered to {len(filtered_data)} high-confidence detections (>= {confidence_threshold}%)")
        logger.info(f"Filtered to {len(filtered_data)} high-confidence detections (>= {confidence_threshold}%)")
        
        # サンプル数制限
        max_samples = self.config['processing']['max_samples']
        if len(filtered_data) > max_samples:
            filtered_data = filtered_data.sample(n=max_samples, random_state=42)
            print(f"Sampled down to {max_samples} records for analysis")
        
        final_data = filtered_data.copy()
        
        logger.info(f"Final dataset: {len(final_data)} NASA FIRMS records for comprehensive analysis")
        
        # データ保存
        data_file = f"{self.output_dir}/nasa_firms_data.csv"
        final_data.to_csv(data_file, index=False)
        logger.info(f"Data saved: {data_file}")
        
        # 統計更新
        self.stats['total_samples'] = len(nasa_data)
        self.stats['final_samples'] = len(final_data)
        
        return {
            'nasa_data': final_data,
            'raw_count': len(nasa_data),
            'filtered_count': len(final_data),
            'confidence_threshold': confidence_threshold
        }
    
    @_time_step("Generating Text Embeddings")
    def _generate_embeddings(self, nasa_data) -> Dict:
        """テキスト埋め込み生成"""
        logger.info("=== Generating Text Embeddings ===")
        
        # 火災検出テキスト記述生成
        text_descriptions = []
        for _, row in nasa_data.iterrows():
            desc = (f"Forest fire detected at latitude {row['latitude']:.4f}, "
                   f"longitude {row['longitude']:.4f} with brightness {row['brightness']:.1f}K "
                   f"and {row['confidence']}% confidence on {row['acq_date']} "
                   f"at {str(row['acq_time']).zfill(4)[:2]}:{str(row['acq_time']).zfill(4)[2:]} "
                   f"by {row['satellite']} {row['instrument']}")
            text_descriptions.append(desc)
        
        logger.info(f"Generated {len(text_descriptions)} fire detection text descriptions")
        
        # 埋め込み生成
        embeddings, scores = self.embedding_generator.generate_embeddings_batch(text_descriptions)
        
        logger.info(f"Generated {len(embeddings)} embeddings (dim: {embeddings.shape[1]})")
        
        # 埋め込み保存
        embeddings_file = f"{self.output_dir}/embeddings.npy"
        np.save(embeddings_file, embeddings.cpu().numpy())
        logger.info(f"Embeddings saved: {embeddings_file}")
        
        return {
            'embeddings': embeddings,
            'scores': scores,
            'text_descriptions': text_descriptions
        }
    
    @_time_step("Performing Adaptive Clustering")
    def _perform_clustering(self, embeddings, nasa_data) -> Dict:
        """適応的クラスタリング実行"""
        logger.info("=== Performing Adaptive Clustering ===")
        
        # 適応的クラスタリング設定
        hdbscan_params = self.config['adaptive_clustering']['hdbscan_params']
        kmeans_params = self.config['adaptive_clustering']['kmeans_params']
        
        logger.info(f"Adaptive parameters: HDBSCAN {hdbscan_params}, k-means {kmeans_params}")
        
        # クラスタリング選択器初期化
        clustering_selector = AdaptiveClusteringSelector(
            output_dir=self.output_dir,
            min_cluster_quality=self.config['adaptive_clustering']['quality_thresholds']['min_cluster_quality'],
            max_noise_ratio=self.config['adaptive_clustering']['quality_thresholds']['max_noise_ratio']
        )
        
        # 最適クラスタリング手法選択・実行
        best_result, selection_details = clustering_selector.select_best_clustering(
            embeddings=embeddings,
            hdbscan_params=hdbscan_params,
            kmeans_params=kmeans_params
        )
        
        # 結果検証
        if best_result is None:
            raise ValueError("Clustering failed - no valid results")
        
        labels = best_result.labels
        method_name = best_result.method
        quality_score = best_result.metrics.quality_score
        
        logger.info(f"Selected method: {method_name}")
        logger.info(f"Selection reason: {selection_details.get('selection_reason', 'Not specified')}")
        
        return {
            'labels': labels,
            'method': method_name,
            'quality_score': quality_score,
            'clustering_result': best_result
        }
    
    @_time_step("Creating Comprehensive Visualizations")
    def _create_visualizations(self, embeddings, labels, scores) -> Dict:
        """包括的可視化作成"""
        logger.info("=== Creating Comprehensive Visualizations ===")
        
        visualization_files = []
        
        # t-SNE次元削減
        coords = self.visualization_manager.reduce_dimensions_tsne(embeddings)
        
        # t-SNE可視化
        tsne_file = self.visualization_manager.create_cluster_plot(
            coords, labels, scores, title="North America FIRMS Fire Analysis Clustering"
        )
        visualization_files.append(tsne_file)
        logger.info(f"✅ Generated t-SNE plot: {tsne_file}")
        
        # スコア分布可視化
        score_file = self.visualization_manager.create_score_distribution_plot(
            scores, labels
        )
        visualization_files.append(score_file)
        logger.info(f"✅ Generated score distribution: {score_file}")
        
        return {
            'visualization_files': visualization_files,
            'tsne_plot': tsne_file,
            'score_distribution': score_file
        }
    
    @_time_step("Performing Cluster Feature Analysis")
    def _analyze_cluster_features(self, nasa_data, labels, embeddings) -> Dict:
        """クラスター特徴分析実行"""
        logger.info("=== Performing Cluster Feature Analysis ===")
        
        # クラスター特徴分析器初期化
        feature_analyzer = ClusterFeatureAnalyzer(output_dir=self.output_dir)
        
        # 包括的特徴分析実行
        feature_analysis = feature_analyzer.analyze_cluster_features(
            nasa_data=nasa_data,
            labels=labels,
            embeddings=embeddings
        )
        
        # 特徴分析可視化作成
        logger.info("Creating feature analysis visualizations...")
        feature_viz_files = feature_analyzer.create_feature_visualizations(
            feature_analysis, self.output_dir
        )
        
        logger.info(f"Created {len(feature_viz_files)} feature visualization files")
        
        # 可視化ファイル表示
        for viz_file in feature_viz_files:
            logger.info(f"  📈 Feature viz: {viz_file}")
        
        return {
            'feature_analysis': feature_analysis,
            'visualization_files': feature_viz_files
        }
    
    @_time_step("Saving Final Results")
    def _save_results(self, nasa_data, labels, embeddings, clustering_result, 
                     feature_analysis, visualizations) -> Dict:
        """最終結果保存"""
        logger.info("=== Saving Final Results ===")
        
        # 統計計算
        elapsed_time = (datetime.now() - self.stats['start_time']).total_seconds()
        
        # クラスタリング結果を辞書形式で作成（EU版と同様）
        clustering_summary = {
            'quality_score': clustering_result.metrics.quality_score,
            'selected_method': clustering_result.method,
            'method_reason': getattr(clustering_result, 'reason', ''),
            'n_clusters': len(set(labels[labels >= 0])),
            'noise_ratio': (labels == -1).sum() / len(labels)
        }
        
        # 最終結果辞書作成
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'region': 'North America',
            'total_samples': self.stats['final_samples'],
            'quality_score': clustering_result.metrics.quality_score,  # レポート用トップレベル
            'clustering': clustering_summary,
            'feature_analysis_summary': {
                'total_clusters': len(set(labels[labels >= 0])),
                'geographic_distribution': len(feature_analysis.get('geographic_analysis', {})),
                'temporal_patterns': len(feature_analysis.get('temporal_analysis', {}))
            },
            'config_used': self.config_file
        }
        
        # 結果ファイル保存
        results_file = f"{self.output_dir}/final_north_america_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final results saved: {results_file}")
        
        # ラベル付きデータ保存
        labeled_data = nasa_data.copy()
        labeled_data['cluster_label'] = labels
        
        labeled_file = f"{self.output_dir}/north_america_fires_clustered.csv"
        labeled_data.to_csv(labeled_file, index=False)
        logger.info(f"Labeled data saved: {labeled_file}")
        
        return {
            'results_file': results_file,
            'labeled_file': labeled_file,
            'final_results': final_results,
            'processing_time': elapsed_time
        }
    
    @_time_step("Generating Comprehensive Analysis Report")
    def _generate_report(self, nasa_data, labels, clustering_result, feature_analysis) -> str:
        """包括的分析レポート生成"""
        logger.info("=== Generating Comprehensive Analysis Report ===")
        
        try:
            # レポート生成器初期化（configを渡す）
            report_generator = FireAnalysisReportGenerator(
                output_dir=self.output_dir, 
                config=self.config
            )
            
            # EU版と同様のレポートデータ構造で作成
            clustering_summary = {
                'quality_score': clustering_result.metrics.quality_score,
                'selected_method': clustering_result.method,
                'method_reason': getattr(clustering_result, 'reason', ''),
                'n_clusters': len(set(labels[labels >= 0])),
                'noise_ratio': (labels == -1).sum() / len(labels)
            }
            
            report_data = {
                'data': nasa_data,
                'labels': labels,
                'clustering_results': clustering_summary,
                'feature_analysis': feature_analysis,
                'region': 'North America',
                'region_name': 'North America',
                'focus_country': 'USA/Canada'
            }
            
            # レポート生成実行
            report_file = report_generator.generate_report(report_data)
            
            logger.info(f"📝 Comprehensive analysis report generated: {report_file}")
            
            return report_file
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            logger.info("Pipeline completed successfully despite report generation issue")
            return None
        
        return report_file
    
    def run_pipeline(self) -> Dict:
        """完全パイプライン実行"""
        try:
            logger.info("🔥 Starting North America FIRMS Fire Analysis Pipeline")
            
            # 1. コンポーネント初期化
            self._initialize_components()
            
            # 2. NASA FIRMSデータ収集
            data_result = self._collect_nasa_firms_data()
            nasa_data = data_result['nasa_data']
            
            # 3. 埋め込み生成
            embedding_result = self._generate_embeddings(nasa_data)
            embeddings = embedding_result['embeddings']
            scores = embedding_result['scores']
            
            # 4. 適応的クラスタリング
            clustering_result = self._perform_clustering(embeddings, nasa_data)
            labels = clustering_result['labels']
            
            # 5. 可視化作成
            visualization_result = self._create_visualizations(embeddings, labels, scores)
            
            # 6. クラスター特徴分析
            feature_result = self._analyze_cluster_features(nasa_data, labels, embeddings)
            feature_analysis = feature_result['feature_analysis']
            
            # 7. 結果保存
            save_result = self._save_results(
                nasa_data, labels, embeddings, clustering_result['clustering_result'],
                feature_analysis, visualization_result
            )
            
            # 8. レポート生成
            report_file = self._generate_report(
                nasa_data, labels, clustering_result['clustering_result'], feature_analysis
            )
            
            # 最終結果まとめ
            pipeline_result = {
                'success': True,
                'output_directory': self.output_dir,
                'total_samples': len(nasa_data),
                'processing_time': save_result['processing_time'],
                'clustering_method': clustering_result['method'],
                'quality_score': clustering_result['quality_score'],
                'n_clusters': len(set(labels[labels >= 0])),
                'noise_ratio': (labels == -1).sum() / len(labels),
                'report_file': report_file,
                'results_file': save_result['results_file']
            }
            
            logger.info("🎉 North America Fire Analysis Pipeline completed successfully!")
            logger.info(f"Processing time: {save_result['processing_time']:.2f}s for {len(nasa_data)} samples")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """メイン実行関数"""
    try:
        # パイプライン初期化・実行
        pipeline = NorthAmericaFIRMSPipeline()
        result = pipeline.run_pipeline()
        
        # 結果表示
        print("\n" + "="*70)
        print("🔥 NORTH AMERICA FOREST FIRE ANALYSIS RESULTS")
        print("="*70)
        print(f"✅ Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"🎯 Selected Method: {result['clustering_method']}")
        print(f"📊 Quality Score: {result['quality_score']:.3f}")
        print(f"🔢 Clusters Found: {result['n_clusters']}")
        print(f"📉 Noise Ratio: {result['noise_ratio']:.1%}")
        print(f"📦 Total Fire Detections: {result['total_samples']}")
        print(f"⏱️ Processing Time: {result['processing_time']:.2f}s")
        print(f"📁 Results Directory: {result['output_directory']}")
        print(f"🌍 Region Coverage: North America (25°N-70°N, 170°W-50°W)")
        print("="*70)
        
        # ファイル一覧表示
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved in: {result['output_directory']}")
        print(f"📄 Main results file: {result['results_file']}")
        
        # 生成ファイル一覧
        import glob
        output_files = glob.glob(f"{result['output_directory']}/*")
        print(f"📊 Generated files ({len(output_files)} total):")
        
        for file_path in sorted(output_files):
            filename = os.path.basename(file_path)
            if filename.endswith('.png'):
                print(f"  🖼️  {filename}")
            elif filename.endswith('.md'):
                print(f"  📝 {filename}", end="")
                if 'comprehensive' in filename:
                    print(" (📖 COMPREHENSIVE ANALYSIS REPORT)")
                else:
                    print()
            elif filename.endswith('.csv'):
                print(f"  📊 {filename}")
            elif filename.endswith('.json'):
                print(f"  📋 {filename}")
            elif filename.endswith('.npy'):
                print(f"  📄 {filename}")
            elif os.path.isdir(file_path):
                print(f"  📄 {filename}")
        
        # 特別レポート案内
        if result.get('report_file') and os.path.exists(result['report_file']):
            print(f"\n📖 **包括的分析レポートが生成されました**")
            print(f"   ファイル: {result['report_file']}")
            print(f"   内容: 6つの図表を用いた詳細な火災分析レポート")
            print(f"   形式: Markdown形式（テキストエディタで閲覧可能）")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()