#!/usr/bin/env python3
"""
HDBSCANクラスタリングシステム
FAISS k-meansの限界を解決する密度ベースクラスタリング
"""

import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import hdbscan
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# v0-7 モジュール
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.append(scripts_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clustering import FastClustering

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日本語フォント設定（Windows対応）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Yu Gothic', 'Meiryo', 'MS Gothic']
plt.rcParams['figure.dpi'] = 100


class HDBSCANClustering:
    """HDBSCAN密度ベースクラスタリング"""
    
    def __init__(self, 
                 min_cluster_size: int = 50,
                 min_samples: int = 10,
                 cluster_selection_epsilon: float = 0.0,
                 metric: str = 'euclidean',
                 cluster_selection_method: str = 'eom',
                 allow_single_cluster: bool = False):
        """
        HDBSCANクラスタリングの初期化
        
        Args:
            min_cluster_size: 最小クラスターサイズ
            min_samples: コアポイントになるための最小サンプル数
            cluster_selection_epsilon: 距離閾値
            metric: 距離メトリック
            cluster_selection_method: クラスター選択方法 ('eom' or 'leaf')
            allow_single_cluster: 単一クラスターを許可するか
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        
        self.clusterer = None
        self.labels_ = None
        self.probabilities_ = None
        self.cluster_persistence_ = None
        self.condensed_tree_ = None
        
    def fit(self, embeddings: np.ndarray) -> 'HDBSCANClustering':
        """HDBSCANクラスタリング実行"""
        logger.info(f"Running HDBSCAN on {embeddings.shape[0]} samples with {embeddings.shape[1]} dimensions")
        
        start_time = time.time()
        
        # データ正規化
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
        
        # HDBSCANクラスタリング
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            prediction_data=True  # 新しいポイントの予測を可能にする
        )
        
        self.labels_ = self.clusterer.fit_predict(normalized_embeddings)
        self.probabilities_ = self.clusterer.probabilities_
        self.cluster_persistence_ = self.clusterer.cluster_persistence_
        self.condensed_tree_ = self.clusterer.condensed_tree_
        
        end_time = time.time()
        
        # 結果統計
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        logger.info(f"HDBSCAN completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Found {n_clusters} clusters with {n_noise} noise points")
        logger.info(f"Noise ratio: {n_noise / len(self.labels_) * 100:.2f}%")
        
        return self
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """クラスター統計情報取得"""
        if self.labels_ is None:
            raise ValueError("Must fit the model first")
        
        unique_labels = set(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.labels_).count(-1)
        
        # クラスターサイズ統計
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:  # ノイズポイントを除外
                cluster_sizes[int(label)] = int(np.sum(self.labels_ == label))
        
        # クラスター永続性統計
        persistence_stats = {}
        if self.cluster_persistence_ is not None:
            if hasattr(self.cluster_persistence_, 'items'):
                # 辞書形式の場合
                for cluster_id, persistence in self.cluster_persistence_.items():
                    persistence_stats[int(cluster_id)] = float(persistence)
            elif isinstance(self.cluster_persistence_, np.ndarray):
                # numpy配列の場合
                for i, persistence in enumerate(self.cluster_persistence_):
                    persistence_stats[i] = float(persistence)
        
        stats = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": n_noise / len(self.labels_),
            "cluster_sizes": cluster_sizes,
            "cluster_persistence": persistence_stats,
            "total_samples": len(self.labels_),
            "parameters": {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "cluster_selection_epsilon": self.cluster_selection_epsilon,
                "metric": self.metric,
                "cluster_selection_method": self.cluster_selection_method
            }
        }
        
        return stats


class HybridClusteringComparator:
    """FAISS k-means vs HDBSCAN 比較システム"""
    
    def __init__(self, output_dir: str = "data_hdbscan_comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def compare_clustering_methods(self, 
                                 embeddings: np.ndarray,
                                 sentiment_scores: np.ndarray,
                                 k_means_clusters: int = 8,
                                 min_cluster_size: int = 50) -> Dict[str, Any]:
        """両クラスタリング手法の比較実行"""
        
        logger.info("=== Clustering Methods Comparison ===")
        
        results = {}
        
        # 1. FAISS k-means クラスタリング
        logger.info("Running FAISS k-means clustering...")
        start_time = time.time()
        
        fast_clustering = FastClustering(output_dir=self.output_dir)
        
        # エンベディングとスコアをtensorに変換
        import torch
        embeddings_tensor = torch.from_numpy(embeddings).float()
        scores_tensor = torch.from_numpy(sentiment_scores).float()
        
        # k-meansクラスタリング実行
        kmeans_labels, centroids = fast_clustering.kmeans_clustering(
            features=embeddings,
            k=k_means_clusters,
            niter=20,
            verbose=True
        )
        
        kmeans_time = time.time() - start_time
        
        # 2. HDBSCAN クラスタリング
        logger.info("Running HDBSCAN clustering...")
        start_time = time.time()
        
        hdbscan_clustering = HDBSCANClustering(
            min_cluster_size=min_cluster_size,
            min_samples=max(10, min_cluster_size // 5),
            metric='euclidean'
        )
        hdbscan_clustering.fit(embeddings)
        hdbscan_labels = hdbscan_clustering.labels_
        
        hdbscan_time = time.time() - start_time
        
        # 3. 評価メトリクス計算
        results = {
            "kmeans": {
                "labels": kmeans_labels.tolist(),
                "n_clusters": k_means_clusters,
                "processing_time": kmeans_time,
                "noise_points": 0,
                "statistics": self._calculate_kmeans_stats(kmeans_labels, sentiment_scores)
            },
            "hdbscan": {
                "labels": hdbscan_labels.tolist(),
                "n_clusters": len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0),
                "processing_time": hdbscan_time,
                "noise_points": int(np.sum(hdbscan_labels == -1)),
                "statistics": hdbscan_clustering.get_cluster_statistics(),
                "probabilities": hdbscan_clustering.probabilities_.tolist() if hdbscan_clustering.probabilities_ is not None else None
            }
        }
        
        # 4. 比較メトリクス
        if len(set(hdbscan_labels)) > 1 and len(set(kmeans_labels)) > 1:
            try:
                # シルエット係数
                kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
                
                # HDBSCANのノイズポイントを除外してシルエット係数計算
                non_noise_mask = hdbscan_labels != -1
                if np.sum(non_noise_mask) > 1 and len(set(hdbscan_labels[non_noise_mask])) > 1:
                    hdbscan_silhouette = silhouette_score(
                        embeddings[non_noise_mask], 
                        hdbscan_labels[non_noise_mask]
                    )
                else:
                    hdbscan_silhouette = 0.0
                
                results["comparison"] = {
                    "kmeans_silhouette_score": float(kmeans_silhouette),
                    "hdbscan_silhouette_score": float(hdbscan_silhouette),
                    "better_silhouette": "HDBSCAN" if hdbscan_silhouette > kmeans_silhouette else "k-means",
                    "performance_comparison": {
                        "kmeans_time": kmeans_time,
                        "hdbscan_time": hdbscan_time,
                        "faster_method": "k-means" if kmeans_time < hdbscan_time else "HDBSCAN"
                    }
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate comparison metrics: {e}")
                results["comparison"] = {"error": str(e)}
        
        # 5. 結果保存
        results_file = os.path.join(self.output_dir, "clustering_comparison_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to {results_file}")
        
        return results
    
    def _calculate_kmeans_stats(self, labels: np.ndarray, sentiment_scores: np.ndarray) -> Dict[str, Any]:
        """k-means統計計算"""
        unique_labels = set(labels)
        cluster_sizes = {}
        cluster_sentiments = {}
        
        for label in unique_labels:
            mask = labels == label
            cluster_sizes[int(label)] = int(np.sum(mask))
            cluster_sentiments[int(label)] = {
                "mean": float(np.mean(sentiment_scores[mask])),
                "std": float(np.std(sentiment_scores[mask])),
                "min": float(np.min(sentiment_scores[mask])),
                "max": float(np.max(sentiment_scores[mask]))
            }
        
        return {
            "cluster_sizes": cluster_sizes,
            "cluster_sentiments": cluster_sentiments,
            "total_samples": len(labels)
        }


def main():
    """テスト実行"""
    # ダミーデータでテスト
    np.random.seed(42)
    n_samples = 1000
    n_features = 128
    
    # 複数のクラスターを持つダミーデータ生成
    embeddings = np.vstack([
        np.random.normal(0, 1, (n_samples//3, n_features)),
        np.random.normal(5, 1.5, (n_samples//3, n_features)),
        np.random.normal(-3, 0.8, (n_samples//3 + n_samples%3, n_features))
    ])
    
    sentiment_scores = np.random.uniform(-1, 1, n_samples)
    
    # 比較実行
    comparator = HybridClusteringComparator()
    results = comparator.compare_clustering_methods(embeddings, sentiment_scores)
    
    print("\n=== HDBSCAN vs k-means Comparison Results ===")
    print(f"k-means clusters: {results['kmeans']['n_clusters']}")
    print(f"HDBSCAN clusters: {results['hdbscan']['n_clusters']}")
    print(f"HDBSCAN noise points: {results['hdbscan']['noise_points']}")
    
    if 'comparison' in results:
        comp = results['comparison']
        print(f"k-means silhouette: {comp.get('kmeans_silhouette_score', 'N/A'):.4f}")
        print(f"HDBSCAN silhouette: {comp.get('hdbscan_silhouette_score', 'N/A'):.4f}")
        print(f"Better method: {comp.get('better_silhouette', 'N/A')}")


if __name__ == "__main__":
    main()