"""
PDF 문장 검색 엔진 - 핵심 기능
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import PyPDF2
import pdfplumber
import re
import nltk
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from datetime import datetime

# NLTK 데이터 다운로드 (필요시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    page_number: int
    similarity_score: float
    start_char: int
    end_char: int
    context_before: str = ""
    context_after: str = ""

class PDFSearchEngine:
    """PDF 문장 검색 엔진"""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 모델 로드
        print(f"🤖 임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # 문서 저장
        self.document_chunks = []
        self.chunk_embeddings = None
        self.pdf_metadata = {}
        
        print(f"✅ 모델 로드 완료 (차원: {self.embedding_dim})")
        
    def load_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 파일 로드 및 처리"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
        print(f"📄 PDF 로딩 중: {pdf_path.name}")
        
        # PDF 메타데이터
        self.pdf_metadata = {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "loaded_at": datetime.now().isoformat()
        }
        
        # PDF 텍스트 추출
        pages_text = self._extract_text_from_pdf(pdf_path)
        
        # 청킹
        self.document_chunks = self._create_chunks(pages_text)
        
        # 임베딩 생성
        self._generate_embeddings()
        
        stats = {
            "total_pages": len(pages_text),
            "total_chunks": len(self.document_chunks),
            "avg_chunk_length": np.mean([len(chunk['content']) for chunk in self.document_chunks]),
            "filename": pdf_path.name
        }
        
        print(f"✅ PDF 처리 완료: {stats['total_pages']}페이지, {stats['total_chunks']}개 청크")
        
        return stats
        
    def _extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """PDF에서 텍스트 추출"""
        pages_text = []
        
        try:
            # pdfplumber 사용 (더 정확한 텍스트 추출)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        pages_text.append({
                            "page_number": page_num,
                            "content": text,
                            "char_count": len(text)
                        })
                        
        except Exception as e:
            # PyPDF2로 fallback
            logger.warning(f"pdfplumber 실패, PyPDF2로 재시도: {e}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        pages_text.append({
                            "page_number": page_num,
                            "content": text,
                            "char_count": len(text)
                        })
                        
        return pages_text
        
    def _create_chunks(self, pages_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """텍스트를 검색 가능한 청크로 분할"""
        chunks = []
        
        for page_data in pages_text:
            page_num = page_data["page_number"]
            text = page_data["content"]
            
            # 문장 분할
            sentences = nltk.sent_tokenize(text)
            
            # 청크 생성
            current_chunk = ""
            current_sentences = []
            
            for sentence in sentences:
                # 현재 청크에 문장 추가 시 크기 확인
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk.split()) <= self.chunk_size:
                    current_chunk = potential_chunk
                    current_sentences.append(sentence)
                else:
                    # 현재 청크 저장
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "page_number": page_num,
                            "sentences": current_sentences.copy(),
                            "word_count": len(current_chunk.split()),
                            "char_start": text.find(current_sentences[0]) if current_sentences else 0,
                            "char_end": text.find(current_sentences[-1]) + len(current_sentences[-1]) if current_sentences else 0
                        })
                    
                    # 새 청크 시작 (오버랩 고려)
                    if self.chunk_overlap > 0 and current_sentences:
                        overlap_sentences = current_sentences[-self.chunk_overlap:]
                        current_chunk = " ".join(overlap_sentences) + " " + sentence
                        current_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence
                        current_sentences = [sentence]
            
            # 마지막 청크 처리
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "page_number": page_num,
                    "sentences": current_sentences,
                    "word_count": len(current_chunk.split()),
                    "char_start": text.find(current_sentences[0]) if current_sentences else 0,
                    "char_end": text.find(current_sentences[-1]) + len(current_sentences[-1]) if current_sentences else 0
                })
                
        return chunks
        
    def _generate_embeddings(self):
        """청크들의 임베딩 생성"""
        if not self.document_chunks:
            return
            
        print(f"🔄 {len(self.document_chunks)}개 청크의 임베딩 생성 중...")
        
        texts = [chunk["content"] for chunk in self.document_chunks]
        self.chunk_embeddings = self.model.encode(texts, 
                                                show_progress_bar=True,
                                                convert_to_numpy=True)
        
        print(f"✅ 임베딩 생성 완료: {self.chunk_embeddings.shape}")
        
    def search(self, 
               query: str, 
               top_k: int = 5,
               min_similarity: float = 0.1) -> List[SearchResult]:
        """문장 검색 수행"""
        
        if not self.document_chunks or self.chunk_embeddings is None:
            raise ValueError("PDF를 먼저 로드해주세요.")
            
        print(f"🔍 검색 중: '{query}'")
        
        # 쿼리 임베딩
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # 유사도 계산
        similarities = self._compute_similarities(query_embedding, self.chunk_embeddings)
        
        # 상위 k개 결과 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            
            if similarity < min_similarity:
                continue
                
            chunk = self.document_chunks[idx]
            
            # 컨텍스트 추가
            context_before, context_after = self._get_context(idx)
            
            result = SearchResult(
                content=chunk["content"],
                page_number=chunk["page_number"],
                similarity_score=float(similarity),
                start_char=chunk.get("char_start", 0),
                end_char=chunk.get("char_end", 0),
                context_before=context_before,
                context_after=context_after
            )
            
            results.append(result)
            
        print(f"✅ {len(results)}개 결과 찾음")
        return results
        
    def _compute_similarities(self, query_embedding: np.ndarray, 
                            doc_embeddings: np.ndarray) -> np.ndarray:
        """코사인 유사도 계산"""
        # 정규화
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # 코사인 유사도
        similarities = np.dot(doc_norms, query_norm)
        return similarities
        
    def _get_context(self, chunk_idx: int, context_size: int = 1) -> Tuple[str, str]:
        """주변 청크를 컨텍스트로 제공"""
        context_before = ""
        context_after = ""
        
        # 이전 청크
        if chunk_idx > 0:
            for i in range(max(0, chunk_idx - context_size), chunk_idx):
                if self.document_chunks[i]["page_number"] == self.document_chunks[chunk_idx]["page_number"]:
                    context_before += self.document_chunks[i]["content"] + " "
                    
        # 다음 청크  
        if chunk_idx < len(self.document_chunks) - 1:
            for i in range(chunk_idx + 1, min(len(self.document_chunks), chunk_idx + context_size + 1)):
                if self.document_chunks[i]["page_number"] == self.document_chunks[chunk_idx]["page_number"]:
                    context_after += self.document_chunks[i]["content"] + " "
                    
        return context_before.strip(), context_after.strip()
        
    def get_statistics(self) -> Dict[str, Any]:
        """문서 통계 정보"""
        if not self.document_chunks:
            return {}
            
        stats = {
            "filename": self.pdf_metadata.get("filename", "Unknown"),
            "total_chunks": len(self.document_chunks),
            "total_pages": max(chunk["page_number"] for chunk in self.document_chunks),
            "avg_chunk_length": np.mean([chunk["word_count"] for chunk in self.document_chunks]),
            "total_words": sum(chunk["word_count"] for chunk in self.document_chunks),
            "pages_distribution": {}
        }
        
        # 페이지별 청크 분포
        for chunk in self.document_chunks:
            page = chunk["page_number"]
            if page not in stats["pages_distribution"]:
                stats["pages_distribution"][page] = 0
            stats["pages_distribution"][page] += 1
            
        return stats
        
    def save_search_results(self, query: str, results: List[SearchResult], 
                          output_path: str = "results/search_results.json"):
        """검색 결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 결과 직렬화
        serialized_results = []
        for result in results:
            serialized_results.append({
                "content": result.content,
                "page_number": result.page_number,
                "similarity_score": result.similarity_score,
                "start_char": result.start_char,
                "end_char": result.end_char,
                "context_before": result.context_before,
                "context_after": result.context_after
            })
            
        # 검색 기록
        search_record = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "pdf_metadata": self.pdf_metadata,
            "results_count": len(results),
            "results": serialized_results
        }
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(search_record, f, indent=2, ensure_ascii=False)
            
        print(f"💾 검색 결과 저장: {output_path}") 