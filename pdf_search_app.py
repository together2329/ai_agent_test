"""
PDF 문장 검색 웹 애플리케이션
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pdf_search_engine import PDFSearchEngine

# 페이지 설정
st.set_page_config(
    page_title="PDF 문장 검색",
    page_icon="🔍",
    layout="wide"
)

# 세션 상태 초기화
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

def load_pdf(uploaded_file):
    """PDF 파일 로드"""
    if uploaded_file is None:
        return None
        
    # 임시 파일 저장
    temp_path = Path("temp") / uploaded_file.name
    temp_path.parent.mkdir(exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # 검색 엔진 초기화
    search_engine = PDFSearchEngine()
    
    try:
        # PDF 로드
        stats = search_engine.load_pdf(str(temp_path))
        return search_engine, stats
    except Exception as e:
        st.error(f"PDF 로드 실패: {str(e)}")
        return None, None
    finally:
        # 임시 파일 삭제
        if temp_path.exists():
            temp_path.unlink()

def display_search_results(results):
    """검색 결과 표시"""
    if not results:
        st.info("검색 결과가 없습니다.")
        return
        
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame([
        {
            "페이지": r.page_number,
            "유사도": f"{r.similarity_score:.3f}",
            "내용": r.content,
            "컨텍스트": f"...{r.context_before} [검색결과] {r.context_after}..."
        }
        for r in results
    ])
    
    # 결과 표시
    st.dataframe(
        results_df,
        column_config={
            "페이지": st.column_config.NumberColumn(width="small"),
            "유사도": st.column_config.TextColumn(width="small"),
            "내용": st.column_config.TextColumn(width="large"),
            "컨텍스트": st.column_config.TextColumn(width="large")
        },
        hide_index=True
    )

def display_statistics(stats):
    """통계 정보 표시"""
    if not stats:
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("총 페이지 수", stats["total_pages"])
        st.metric("총 청크 수", stats["total_chunks"])
        
    with col2:
        st.metric("평균 청크 길이", f"{stats['avg_chunk_length']:.1f} 단어")
        st.metric("총 단어 수", stats["total_words"])
        
    # 페이지별 청크 분포 시각화
    pages_dist = stats["pages_distribution"]
    fig = px.bar(
        x=list(pages_dist.keys()),
        y=list(pages_dist.values()),
        labels={"x": "페이지", "y": "청크 수"},
        title="페이지별 청크 분포"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_search_history():
    """검색 기록 표시"""
    if not st.session_state.search_history:
        return
        
    st.subheader("검색 기록")
    
    for query, timestamp in st.session_state.search_history:
        st.text(f"{timestamp}: {query}")

# 메인 UI
st.title("🔍 PDF 문장 검색")

# 사이드바
with st.sidebar:
    st.header("PDF 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf")
    
    if uploaded_file:
        if st.session_state.current_pdf != uploaded_file.name:
            with st.spinner("PDF 로딩 중..."):
                search_engine, stats = load_pdf(uploaded_file)
                if search_engine:
                    st.session_state.search_engine = search_engine
                    st.session_state.current_pdf = uploaded_file.name
                    st.success(f"PDF 로드 완료: {uploaded_file.name}")
                    
                    # 통계 정보 표시
                    st.header("문서 통계")
                    display_statistics(stats)
    
    st.divider()
    display_search_history()

# 메인 영역
if st.session_state.search_engine:
    # 검색 폼
    with st.form("search_form"):
        query = st.text_input("검색어를 입력하세요")
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("결과 개수", 1, 20, 5)
            
        with col2:
            min_similarity = st.slider("최소 유사도", 0.0, 1.0, 0.1, 0.1)
            
        submitted = st.form_submit_button("검색")
        
        if submitted and query:
            with st.spinner("검색 중..."):
                # 검색 수행
                results = st.session_state.search_engine.search(
                    query=query,
                    top_k=top_k,
                    min_similarity=min_similarity
                )
                
                # 검색 기록 추가
                st.session_state.search_history.append((
                    query,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                
                # 결과 저장
                st.session_state.search_engine.save_search_results(
                    query=query,
                    results=results
                )
                
                # 결과 표시
                st.subheader(f"검색 결과 ({len(results)}개)")
                display_search_results(results)
else:
    st.info("�� PDF 파일을 업로드해주세요.") 