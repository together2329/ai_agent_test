#!/usr/bin/env python3
"""
고급 Embedding 기능 테스트 (Docker용)
- 결과 저장 기능 추가
- 시각화 포함
- 상세한 분석
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import os
from datetime import datetime
from pathlib import Path

# 결과 저장 디렉토리
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerEmbeddingTester:
    """Docker 환경용 확장된 임베딩 테스터"""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 batch_size: int = 32):
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.test_results = {}
        
        # 디바이스 설정
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # 모델 로드
        print("🤖 모델 로딩 중...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"✅ 모델 로드 완료: {model_name}")
        logger.info(f"📐 임베딩 차원: {self.embedding_dimension}")
        
    def run_comprehensive_tests(self):
        """전체 테스트 suite 실행"""
        print("\n🧪 종합 임베딩 테스트 시작")
        print("="*50)
        
        # 각 테스트 실행
        self.test_single_embedding()
        self.test_batch_embedding()
        self.test_uvm_error_analysis()
        self.test_similarity_search()
        self.benchmark_performance()
        self.generate_visualizations()
        
        # 결과 저장
        self.save_results()
        
        print("\n🎉 모든 테스트 완료!")
        print(f"📁 결과는 {RESULTS_DIR} 디렉토리에서 확인하세요.")
    
    def test_single_embedding(self):
        """단일 텍스트 임베딩 테스트"""
        print("\n=== 단일 텍스트 임베딩 테스트 ===")
        
        test_text = "UVM_ERROR: Sequence timeout occurred in testbench component"
        embedding = self.model.encode(test_text, convert_to_numpy=True)
        
        results = {
            "input_text": test_text,
            "embedding_shape": embedding.shape,
            "embedding_dtype": str(embedding.dtype),
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding_mean": float(np.mean(embedding)),
            "embedding_std": float(np.std(embedding)),
            "first_5_values": embedding[:5].tolist()
        }
        
        self.test_results["single_embedding"] = results
        
        print(f"✅ 입력: {test_text}")
        print(f"📊 임베딩 shape: {embedding.shape}")
        print(f"📈 L2 norm: {results['embedding_norm']:.4f}")
        
    def test_batch_embedding(self):
        """배치 임베딩 테스트"""
        print("\n=== 배치 임베딩 테스트 ===")
        
        # 다양한 UVM 에러들
        test_texts = [
            "UVM_ERROR: Sequence timeout in driver component",
            "UVM_FATAL: Null pointer access in scoreboard", 
            "UVM_WARNING: Phase timeout in test sequence",
            "UVM_ERROR: Protocol violation detected in monitor",
            "UVM_ERROR: Transaction mismatch in comparator",
            "UVM_FATAL: Memory allocation failed in testbench",
            "UVM_WARNING: Clock frequency mismatch detected",
            "UVM_ERROR: Interface handshake failure",
            "UVM_FATAL: Configuration object not found",
            "UVM_WARNING: Deprecated function usage detected"
        ]
        
        embeddings = self.embed_texts(test_texts, show_progress=True)
        
        # 통계 계산
        results = {
            "num_texts": len(test_texts),
            "embedding_shape": embeddings.shape,
            "texts": test_texts,
            "embedding_stats": {
                "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
                "dimension": int(embeddings.shape[1])
            }
        }
        
        self.test_results["batch_embedding"] = results
        
        print(f"✅ 처리된 텍스트: {len(test_texts)}개")
        print(f"📊 임베딩 배열: {embeddings.shape}")
        
    def test_uvm_error_analysis(self):
        """UVM 에러 분석 테스트"""
        print("\n=== UVM 에러 패턴 분석 ===")
        
        # 에러 유형별 샘플
        error_categories = {
            "Timeout": [
                "UVM_ERROR: Sequence timeout in driver",
                "UVM_WARNING: Phase timeout detected", 
                "UVM_ERROR: Transaction timeout occurred"
            ],
            "Protocol": [
                "UVM_ERROR: Protocol violation in handshake",
                "UVM_ERROR: Interface protocol mismatch",
                "UVM_FATAL: Protocol error in bus transaction"
            ],
            "Memory": [
                "UVM_FATAL: Null pointer exception",
                "UVM_FATAL: Memory allocation failed",
                "UVM_ERROR: Buffer overflow detected"
            ],
            "Configuration": [
                "UVM_ERROR: Configuration object not found",
                "UVM_WARNING: Invalid configuration parameter",
                "UVM_FATAL: Configuration database corrupted"
            ]
        }
        
        category_embeddings = {}
        
        for category, texts in error_categories.items():
            embeddings = self.embed_texts(texts, show_progress=False)
            category_embeddings[category] = embeddings
            
            # 카테고리 내 평균 임베딩
            avg_embedding = np.mean(embeddings, axis=0)
            
            print(f"📂 {category}: {len(texts)}개 에러")
            
        # 카테고리 간 유사도 계산
        category_similarities = self.compute_category_similarities(category_embeddings)
        
        self.test_results["uvm_error_analysis"] = {
            "categories": list(error_categories.keys()),
            "category_similarities": category_similarities,
            "error_samples": error_categories
        }
        
    def test_similarity_search(self):
        """유사도 검색 성능 테스트"""
        print("\n=== 유사도 검색 테스트 ===")
        
        # UVM 문서 데이터베이스
        uvm_docs = [
            "UVM sequence generates constrained random stimulus for verification",
            "Driver component sends transactions to DUT through virtual interface",
            "Monitor passively observes interface signals and creates transactions", 
            "Scoreboard compares predicted and actual DUT responses",
            "Agent encapsulates driver, monitor, and sequencer in reusable unit",
            "Factory enables runtime creation and configuration of UVM objects",
            "Phasing provides synchronization and ordering of testbench activities",
            "Configuration database stores and retrieves testbench parameters",
            "Transaction class models protocol-specific data and operations",
            "Virtual interface provides abstraction layer for signal connections"
        ]
        
        doc_embeddings = self.embed_texts(uvm_docs, show_progress=False)
        
        # 테스트 쿼리들
        test_queries = [
            "sequence timeout problem",
            "driver interface error",
            "scoreboard comparison failure",
            "monitor sampling issue"
        ]
        
        search_results = {}
        
        for query in test_queries:
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            similarities = self.compute_similarity(query_embedding, doc_embeddings)
            
            # 상위 3개 결과
            top_indices = np.argsort(similarities)[::-1][:3]
            
            results = []
            for idx in top_indices:
                results.append({
                    "document": uvm_docs[idx],
                    "similarity": float(similarities[idx])
                })
                
            search_results[query] = results
            
            print(f"🔍 '{query}' 검색 결과:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. ({result['similarity']:.4f}) {result['document'][:50]}...")
                
        self.test_results["similarity_search"] = search_results
        
    def benchmark_performance(self):
        """성능 벤치마크"""
        print("\n=== 성능 벤치마크 ===")
        
        import time
        
        # 테스트 데이터 생성
        test_sizes = [10, 50, 100, 500, 1000]
        performance_results = {}
        
        for size in test_sizes:
            test_texts = [f"UVM test sentence number {i} for performance evaluation." 
                         for i in range(size)]
            
            start_time = time.time()
            embeddings = self.embed_texts(test_texts, show_progress=False)
            end_time = time.time()
            
            elapsed = end_time - start_time
            throughput = size / elapsed
            
            performance_results[size] = {
                "elapsed_time": elapsed,
                "throughput": throughput,
                "texts_per_second": throughput
            }
            
            print(f"📊 {size:4d} 텍스트: {elapsed:.2f}초, {throughput:.1f} 텍스트/초")
            
        self.test_results["performance"] = performance_results
        
    def generate_visualizations(self):
        """시각화 생성"""
        print("\n=== 시각화 생성 중 ===")
        
        # 1. 성능 차트
        self.plot_performance()
        
        # 2. 임베딩 분포 히트맵  
        self.plot_embedding_heatmap()
        
        # 3. 유사도 매트릭스
        self.plot_similarity_matrix()
        
        print("📈 시각화 완료!")
        
    def plot_performance(self):
        """성능 벤치마크 시각화"""
        if "performance" not in self.test_results:
            return
            
        perf_data = self.test_results["performance"]
        sizes = list(perf_data.keys())
        throughputs = [perf_data[size]["throughput"] for size in sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, throughputs, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Texts')
        plt.ylabel('Throughput (texts/second)')
        plt.title('Embedding Performance Benchmark')
        plt.grid(True, alpha=0.3)
        plt.savefig(RESULTS_DIR / 'performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_embedding_heatmap(self):
        """임베딩 분포 히트맵"""
        if "batch_embedding" not in self.test_results:
            return
            
        # 배치 임베딩 데이터 재생성 (시각화용)
        test_texts = self.test_results["batch_embedding"]["texts"]
        embeddings = self.embed_texts(test_texts, show_progress=False)
        
        # 첫 50개 차원만 시각화
        plt.figure(figsize=(12, 8))
        sns.heatmap(embeddings[:, :50], 
                   cmap='coolwarm', 
                   center=0,
                   xticklabels=False,
                   yticklabels=[f"Text {i+1}" for i in range(len(test_texts))])
        plt.title('Embedding Distribution Heatmap (First 50 dimensions)')
        plt.xlabel('Embedding Dimensions')
        plt.ylabel('Input Texts')
        plt.savefig(RESULTS_DIR / 'embedding_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_similarity_matrix(self):
        """유사도 매트릭스 시각화"""
        if "batch_embedding" not in self.test_results:
            return
            
        test_texts = self.test_results["batch_embedding"]["texts"]
        embeddings = self.embed_texts(test_texts, show_progress=False)
        
        # 유사도 매트릭스 계산
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   square=True,
                   xticklabels=[f"T{i+1}" for i in range(len(test_texts))],
                   yticklabels=[f"T{i+1}" for i in range(len(test_texts))])
        plt.title('Text Similarity Matrix')
        plt.xlabel('Texts')
        plt.ylabel('Texts')
        plt.savefig(RESULTS_DIR / 'similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_category_similarities(self, category_embeddings: Dict[str, np.ndarray]) -> Dict:
        """카테고리 간 유사도 계산"""
        categories = list(category_embeddings.keys())
        similarities = {}
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i <= j:
                    continue
                    
                # 각 카테고리의 평균 임베딩
                avg1 = np.mean(category_embeddings[cat1], axis=0)
                avg2 = np.mean(category_embeddings[cat2], axis=0)
                
                # 코사인 유사도
                similarity = np.dot(avg1, avg2) / (np.linalg.norm(avg1) * np.linalg.norm(avg2))
                similarities[f"{cat1}-{cat2}"] = float(similarity)
                
        return similarities
        
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """배치 텍스트 임베딩"""
        if not texts:
            return np.array([])
            
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     disable=not show_progress,
                     desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """코사인 유사도 계산"""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        return similarities
        
    def save_results(self):
        """테스트 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        results_file = RESULTS_DIR / f"embedding_test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
        # 요약 리포트 생성
        self.generate_summary_report(timestamp)
        
        print(f"💾 결과 저장 완료: {results_file}")
        
    def generate_summary_report(self, timestamp: str):
        """요약 리포트 생성"""
        report_file = RESULTS_DIR / f"embedding_test_summary_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Embedding Test Summary Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model**: {self.model_name}\n")
            f.write(f"**Device**: {self.device}\n")
            f.write(f"**Embedding Dimension**: {self.embedding_dimension}\n\n")
            
            # 성능 요약
            if "performance" in self.test_results:
                f.write("## 성능 벤치마크\n\n")
                perf_data = self.test_results["performance"]
                for size, metrics in perf_data.items():
                    f.write(f"- {size} texts: {metrics['throughput']:.1f} texts/sec\n")
                f.write("\n")
            
            # UVM 에러 분석 요약
            if "uvm_error_analysis" in self.test_results:
                f.write("## UVM 에러 분석\n\n")
                categories = self.test_results["uvm_error_analysis"]["categories"]
                f.write(f"분석된 에러 카테고리: {', '.join(categories)}\n\n")
            
            f.write("## 파일 목록\n\n")
            f.write("- `embedding_test_results_*.json`: 상세 테스트 결과\n")
            f.write("- `performance_benchmark.png`: 성능 벤치마크 차트\n")
            f.write("- `embedding_heatmap.png`: 임베딩 분포 히트맵\n")
            f.write("- `similarity_matrix.png`: 텍스트 유사도 매트릭스\n")

def main():
    """메인 함수"""
    print("🐳 Docker 환경 Embedding 테스트")
    print("=" * 40)
    
    # 환경 정보 출력
    print(f"🐍 Python version: {torch.__version__}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU device: {torch.cuda.get_device_name(0)}")
    
    # 테스터 실행
    tester = DockerEmbeddingTester()
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main() 