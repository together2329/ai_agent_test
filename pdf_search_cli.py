"""
PDF 문장 검색 CLI
"""

import argparse
import json
from pathlib import Path
from pdf_search_engine import PDFSearchEngine
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

console = Console()

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="PDF 문장 검색 CLI")
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="검색할 PDF 파일 경로"
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="검색어 (미지정시 대화형 모드)"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="반환할 결과 개수 (기본값: 5)"
    )
    
    parser.add_argument(
        "-s", "--min-similarity",
        type=float,
        default=0.1,
        help="최소 유사도 (기본값: 0.1)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="결과 저장 경로 (JSON)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="문서 통계 정보 표시"
    )
    
    return parser.parse_args()

def display_results(results, show_context=True):
    """검색 결과 표시"""
    if not results:
        console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        return
        
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("페이지", style="dim")
    table.add_column("유사도", justify="right")
    table.add_column("내용")
    
    if show_context:
        table.add_column("컨텍스트")
        
    for result in results:
        row = [
            str(result.page_number),
            f"{result.similarity_score:.3f}",
            result.content
        ]
        
        if show_context:
            context = f"...{result.context_before} [검색결과] {result.context_after}..."
            row.append(context)
            
        table.add_row(*row)
        
    console.print(table)

def display_stats(stats):
    """통계 정보 표시"""
    if not stats:
        return
        
    console.print("\n[bold cyan]문서 통계[/bold cyan]")
    
    # 기본 통계
    console.print(f"파일명: {stats['filename']}")
    console.print(f"총 페이지 수: {stats['total_pages']}")
    console.print(f"총 청크 수: {stats['total_chunks']}")
    console.print(f"평균 청크 길이: {stats['avg_chunk_length']:.1f} 단어")
    console.print(f"총 단어 수: {stats['total_words']}")
    
    # 페이지별 청크 분포
    console.print("\n[bold]페이지별 청크 분포[/bold]")
    for page, count in sorted(stats["pages_distribution"].items()):
        console.print(f"페이지 {page}: {count}개 청크")

def interactive_mode(search_engine):
    """대화형 검색 모드"""
    console.print("[bold green]대화형 검색 모드[/bold green]")
    console.print("종료하려면 'q' 또는 'quit'를 입력하세요.")
    
    while True:
        try:
            query = console.input("\n[bold cyan]검색어: [/bold cyan]")
            
            if query.lower() in ['q', 'quit']:
                break
                
            if not query.strip():
                continue
                
            with Progress() as progress:
                task = progress.add_task("[cyan]검색 중...", total=None)
                results = search_engine.search(query)
                
            display_results(results)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]오류: {str(e)}[/red]")

def main():
    """메인 함수"""
    args = parse_args()
    
    # PDF 파일 확인
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        console.print(f"[red]오류: PDF 파일을 찾을 수 없습니다: {pdf_path}[/red]")
        sys.exit(1)
        
    try:
        # 검색 엔진 초기화
        with Progress() as progress:
            task = progress.add_task("[cyan]PDF 로딩 중...", total=None)
            search_engine = PDFSearchEngine()
            stats = search_engine.load_pdf(str(pdf_path))
            
        # 통계 정보 표시
        if args.stats:
            display_stats(stats)
            
        # 검색 수행
        if args.query:
            # 단일 검색
            results = search_engine.search(
                query=args.query,
                top_k=args.top_k,
                min_similarity=args.min_similarity
            )
            
            # 결과 표시
            display_results(results)
            
            # 결과 저장
            if args.output:
                search_engine.save_search_results(
                    query=args.query,
                    results=results,
                    output_path=args.output
                )
                console.print(f"[green]결과가 저장되었습니다: {args.output}[/green]")
        else:
            # 대화형 모드
            interactive_mode(search_engine)
            
    except Exception as e:
        console.print(f"[red]오류: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 