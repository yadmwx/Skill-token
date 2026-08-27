#!/bin/bash
# Linux/macOS 定时任务设置脚本
# 使用 crontab 设置每日任务

SCRIPT_PATH="/path/to/vla-flow/scripts/run_tracker.sh"
CRON_JOB="0 8 * * * $SCRIPT_PATH >> /tmp/arxiv_tracker.log 2>&1"

# 检查是否已存在
if crontab -l 2>/dev/null | grep -q "arxiv_tracker"; then
    echo "定时任务已存在，正在更新..."
    # 移除旧任务
    crontab -l 2>/dev/null | grep -v "arxiv_tracker" | crontab -
fi

# 添加新任务
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "已创建定时任务: 每天 8:00 运行"
echo "日志文件: /tmp/arxiv_tracker.log"