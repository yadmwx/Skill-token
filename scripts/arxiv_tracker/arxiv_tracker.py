#!/usr/bin/env python3
"""
arXiv 论文自动追踪系统
每天搜索并下载 VLA/机器人操作相关论文，生成研究进展报告
"""

import arxiv
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import pytz
import re
import ssl
import urllib.request

# 解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# ==================== 配置 ====================

# 项目根目录 (脚本在 scripts/arxiv_tracker/ 下，需要往上两级)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 搜索关键词配置
SEARCH_QUERIES = [
    # VLA 相关
    'Vision-Language-Action robot',
    'VLA robot manipulation',
    # 机器人策略学习
    'robot policy learning manipulation',
    'robot manipulation diffusion policy',
    'robot imitation learning',
    # 具身智能
    'embodied AI robot',
    'robot learning control',
]

# 关键词过滤（必须包含至少一个）
KEYWORD_FILTERS = [
    'VLA', 'vision-language-action', 'robot manipulation',
    'robot policy', 'diffusion policy', 'imitation learning',
    'robot learning', 'embodied', 'manipulation', 'action',
    'robotics', 'robot control', 'policy gradient', 'reinforcement',
    'flow matching', 'action prediction', 'trajectory'
]

# 排除关键词
EXCLUDE_KEYWORDS = [
    'survey', 'review', 'tutorial', 'introduction to',
    'benchmark only', 'dataset only'
]

# 每天下载数量
PAPERS_PER_DAY = 3

# 搜索天数范围（最近N天）
SEARCH_DAYS = 7

# ==================== 核心功能 ====================

class ArxivTracker:
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or PROJECT_ROOT
        # 确保使用正确的路径
        self.paper_dir = Path(self.project_root) / "paper" / "auto_downloaded"
        self.report_dir = Path(self.project_root) / "paper" / "daily_reports"
        self.ui_data_file = Path(self.project_root) / "paper" / "ui" / "data.json"
        self.history_file = Path(self.project_root) / "scripts" / "arxiv_tracker" / "download_history.json"

        # 创建目录
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 加载历史记录
        self.history = self._load_history()

        # arxiv 客户端
        self.client = arxiv.Client()

    def _load_history(self) -> Dict:
        """加载下载历史"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"downloaded": [], "reports": []}

    def _save_history(self):
        """保存下载历史"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _get_arxiv_id(self, paper: arxiv.Result) -> str:
        """从 entry_id 提取 arxiv_id"""
        # entry_id 格式: http://arxiv.org/abs/2301.12345v1
        match = re.search(r'(\d{4}\.\d{4,5}(v\d+)?)', paper.entry_id)
        if match:
            return match.group(1)
        return paper.entry_id.split('/')[-1]

    def _is_relevant(self, paper: arxiv.Result) -> bool:
        """检查论文是否相关"""
        text = f"{paper.title} {paper.summary}".lower()

        # 检查排除关键词
        for kw in EXCLUDE_KEYWORDS:
            if kw.lower() in text:
                return False

        # 检查必须包含的关键词
        keyword_count = sum(1 for kw in KEYWORD_FILTERS if kw.lower() in text)
        return keyword_count >= 2

    def _calculate_relevance_score(self, paper: arxiv.Result) -> float:
        """计算论文相关性分数"""
        text = f"{paper.title} {paper.summary}".lower()
        score = 0.0

        # 高权重关键词
        high_weight = ['vla', 'vision-language-action', 'diffusion policy', 'flow matching']
        for kw in high_weight:
            if kw.lower() in text:
                score += 3.0

        # 中权重关键词
        mid_weight = ['robot manipulation', 'robot policy', 'imitation learning', 'embodied']
        for kw in mid_weight:
            if kw.lower() in text:
                score += 2.0

        # 普通关键词
        for kw in KEYWORD_FILTERS:
            if kw.lower() in text:
                score += 1.0

        # 新鲜度加分
        days_old = (datetime.now(pytz.UTC) - paper.published).days
        if days_old <= 1:
            score += 2.0
        elif days_old <= 3:
            score += 1.0

        return score

    def search_papers(self, max_results: int = 50) -> List[arxiv.Result]:
        """搜索最新论文"""
        all_papers = []

        for query in SEARCH_QUERIES:
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )

                for paper in self.client.results(search):
                    # 获取 arxiv_id
                    arxiv_id = self._get_arxiv_id(paper)

                    # 检查是否已下载
                    if arxiv_id in self.history["downloaded"]:
                        continue

                    # 检查时间范围
                    days_old = (datetime.now(pytz.UTC) - paper.published).days
                    if days_old > SEARCH_DAYS:
                        continue

                    # 检查相关性
                    if self._is_relevant(paper):
                        all_papers.append(paper)

            except Exception as e:
                print(f"搜索 '{query}' 时出错: {e}")

        # 去重并排序
        seen = set()
        unique_papers = []
        for paper in all_papers:
            arxiv_id = self._get_arxiv_id(paper)
            if arxiv_id not in seen:
                seen.add(arxiv_id)
                unique_papers.append(paper)

        # 按相关性排序
        unique_papers.sort(key=self._calculate_relevance_score, reverse=True)

        return unique_papers

    def download_paper(self, paper: arxiv.Result) -> Path:
        """下载论文 PDF"""
        arxiv_id = self._get_arxiv_id(paper)
        date_str = datetime.now().strftime("%Y-%m-%d")
        download_dir = self.paper_dir / date_str
        download_dir.mkdir(parents=True, exist_ok=True)

        # 安全的文件名
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)
        safe_title = safe_title[:80]  # 限制长度
        filename = f"{arxiv_id.replace('/', '_')}_{safe_title}.pdf"
        filepath = download_dir / filename

        if not filepath.exists():
            print(f"正在下载: {paper.title}")
            paper.download_pdf(dirpath=str(download_dir), filename=filename)
        else:
            print(f"已存在: {paper.title}")

        return filepath

    def generate_report(self, papers: List[arxiv.Result]) -> Path:
        """生成每日报告"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = self.report_dir / f"report_{date_str}.md"

        # 分类论文
        categories = {
            "VLA/视觉-语言-动作": [],
            "机器人策略学习": [],
            "具身智能": [],
            "其他": []
        }

        for paper in papers:
            text = f"{paper.title} {paper.summary}".lower()
            if 'vla' in text or 'vision-language-action' in text:
                categories["VLA/视觉-语言-动作"].append(paper)
            elif 'policy' in text or 'diffusion' in text or 'imitation' in text:
                categories["机器人策略学习"].append(paper)
            elif 'embodied' in text or 'robot learning' in text:
                categories["具身智能"].append(paper)
            else:
                categories["其他"].append(paper)

        # 生成报告
        report_content = f"""# arXiv 每日论文报告

**日期**: {date_str}
**论文数量**: {len(papers)}

---

## 摘要

本报告汇总了最近 {SEARCH_DAYS} 天内 arXiv 上关于 VLA/机器人操作的最新研究进展。

---

## 论文列表

"""

        for category, category_papers in categories.items():
            if not category_papers:
                continue

            report_content += f"### {category} ({len(category_papers)} 篇)\n\n"

            for i, paper in enumerate(category_papers, 1):
                arxiv_id = self._get_arxiv_id(paper)
                score = self._calculate_relevance_score(paper)
                authors = ", ".join(a.name for a in paper.authors[:3])
                if len(paper.authors) > 3:
                    authors += " et al."

                report_content += f"""#### {i}. {paper.title}

- **arXiv ID**: [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})
- **作者**: {authors}
- **发布日期**: {paper.published.strftime('%Y-%m-%d')}
- **相关性评分**: {score:.1f}

**摘要**:
{paper.summary[:500]}{'...' if len(paper.summary) > 500 else ''}

---

"""

        # 添加研究趋势分析
        report_content += """## 研究趋势分析

### 热门方向
"""
        trend_counts = {k: len(v) for k, v in categories.items() if len(v) > 0}
        for topic, count in sorted(trend_counts.items(), key=lambda x: x[1], reverse=True):
            report_content += f"- {topic}: {count} 篇论文\n"

        report_content += """
### 建议

"""
        if len(papers) > 0:
            top_paper = papers[0]
            top_arxiv_id = self._get_arxiv_id(top_paper)
            report_content += f"**重点关注**: [{top_paper.title}](https://arxiv.org/abs/{top_arxiv_id}) - 相关性评分最高\n"

        report_content += """
---

*报告由 VLA-Flow 自动化论文追踪系统生成*
"""

        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"报告已生成: {report_path}")
        return report_path

    def update_ui_data(self, papers: List[arxiv.Result]):
        """更新 UI 数据文件"""
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 加载现有数据
        existing_data = {"lastUpdated": date_str, "todayPapers": [], "dailyBriefs": []}
        if self.ui_data_file.exists():
            try:
                with open(self.ui_data_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                pass

        # 更新今日论文
        today_papers = []
        for paper in papers[:5]:  # 只取前5篇
            arxiv_id = self._get_arxiv_id(paper)
            score = self._calculate_relevance_score(paper)
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."

            # 判断分类
            text = f"{paper.title} {paper.summary}".lower()
            if 'vla' in text or 'vision-language-action' in text:
                category = "VLA/视觉-语言-动作"
            elif 'policy' in text or 'diffusion' in text or 'imitation' in text:
                category = "机器人策略学习"
            elif 'embodied' in text or 'robot learning' in text:
                category = "具身智能"
            else:
                category = "其他"

            today_papers.append({
                "id": arxiv_id,
                "title": paper.title,
                "authors": authors,
                "score": score,
                "date": paper.published.strftime('%Y-%m-%d'),
                "abstract": paper.summary[:200] + "..." if len(paper.summary) > 200 else paper.summary,
                "category": category
            })

        # 生成今日简报
        top_keyword = "研究进展"
        if today_papers:
            # 从第一篇论文标题提取关键词
            first_title = today_papers[0]['title']
            if ':' in first_title:
                top_keyword = first_title.split(':')[0].split()[-1]
            else:
                top_keyword = first_title.split()[0]

        today_brief = {
            "date": date_str,
            "papers": len(papers),
            "topKeyword": top_keyword,
            "summary": self._generate_summary(papers)
        }

        # 更新简报列表（保持最新在前，避免重复）
        daily_briefs = existing_data.get("dailyBriefs", [])
        # 检查是否已有今天的简报
        existing_dates = [b["date"] for b in daily_briefs]
        if date_str in existing_dates:
            # 更新今天的简报
            daily_briefs = [today_brief if b["date"] == date_str else b for b in daily_briefs]
        else:
            # 添加新简报到前面
            daily_briefs = [today_brief] + daily_briefs

        # 只保留最近10条
        daily_briefs = daily_briefs[:10]

        # 构建新数据
        ui_data = {
            "lastUpdated": date_str,
            "todayPapers": today_papers,
            "dailyBriefs": daily_briefs
        }

        # 保存
        self.ui_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ui_data_file, 'w', encoding='utf-8') as f:
            json.dump(ui_data, f, ensure_ascii=False, indent=2)

        print(f"UI 数据已更新: {self.ui_data_file}")

    def _generate_summary(self, papers: List[arxiv.Result]) -> str:
        """生成简短的趋势总结"""
        if not papers:
            return "暂无明显趋势"

        # 统计关键词
        all_text = " ".join([f"{p.title} {p.summary}" for p in papers]).lower()

        trends = []
        if 'memory' in all_text or 'recurrent' in all_text:
            trends.append("记忆机制")
        if 'flow matching' in all_text or 'flow-matching' in all_text:
            trends.append("Flow Matching")
        if 'diffusion' in all_text:
            trends.append("扩散策略")
        if 'continual' in all_text:
            trends.append("持续学习")
        if 'manipulation' in all_text:
            trends.append("操作策略")

        if trends:
            return "、".join(trends[:2]) + "成为研究热点"
        return "VLA相关研究持续推进"

    def run(self, download: bool = True, report: bool = True, papers_count: int = None):
        """运行追踪系统"""
        papers_count = papers_count or PAPERS_PER_DAY

        print("=" * 60)
        print("arXiv 论文追踪系统")
        print("=" * 60)
        print(f"搜索关键词: {len(SEARCH_QUERIES)} 个查询")
        print(f"目标下载数量: {papers_count} 篇")
        print(f"搜索时间范围: 最近 {SEARCH_DAYS} 天")
        print("=" * 60)

        # 搜索论文
        print("\n正在搜索论文...")
        papers = self.search_papers()
        print(f"找到 {len(papers)} 篇相关论文")

        if not papers:
            print("未找到新的相关论文")
            return []

        # 显示前5篇
        print("\n相关性最高的论文:")
        for i, paper in enumerate(papers[:5], 1):
            score = self._calculate_relevance_score(paper)
            arxiv_id = self._get_arxiv_id(paper)
            print(f"  {i}. [{score:.1f}] [{arxiv_id}] {paper.title[:50]}...")

        # 下载论文
        downloaded_papers = []
        if download:
            print(f"\n正在下载前 {papers_count} 篇论文...")
            for paper in papers[:papers_count]:
                filepath = self.download_paper(paper)
                arxiv_id = self._get_arxiv_id(paper)
                downloaded_papers.append({
                    "id": arxiv_id,
                    "title": paper.title,
                    "path": str(filepath),
                    "downloaded_at": datetime.now().isoformat()
                })
                self.history["downloaded"].append(arxiv_id)

        # 生成报告
        if report:
            print("\n正在生成报告...")
            report_path = self.generate_report(papers[:papers_count])
            self.history["reports"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "path": str(report_path),
                "papers_count": len(papers[:papers_count])
            })

        # 更新 UI 数据
        print("\n正在更新 UI 数据...")
        self.update_ui_data(papers[:papers_count])

        # 保存历史
        self._save_history()

        print("\n" + "=" * 60)
        print("完成!")
        print(f"下载论文: {len(downloaded_papers)} 篇")
        print("=" * 60)

        return downloaded_papers


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='arXiv 论文自动追踪系统')
    parser.add_argument('--no-download', action='store_true', help='不下载 PDF，仅生成报告')
    parser.add_argument('--no-report', action='store_true', help='不生成报告')
    parser.add_argument('-n', '--count', type=int, default=PAPERS_PER_DAY, help='下载数量')
    parser.add_argument('--project-root', type=str, help='项目根目录')

    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT

    tracker = ArxivTracker(project_root)
    tracker.run(
        download=not args.no_download,
        report=not args.no_report,
        papers_count=args.count
    )


if __name__ == "__main__":
    main()