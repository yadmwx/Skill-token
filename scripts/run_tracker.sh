#!/bin/bash
# arXiv 论文追踪系统 - 快捷运行脚本
# 用法: ./run_tracker.sh [选项]

cd "$(dirname "$0")/.."
python scripts/arxiv_tracker/arxiv_tracker.py "$@"