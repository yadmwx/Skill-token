@echo off
REM arXiv 论文追踪系统 - 快捷运行脚本 (Windows)
REM 用法: run_tracker.bat [选项]

cd /d "%~dp0.."
python scripts/arxiv_tracker/arxiv_tracker.py %*