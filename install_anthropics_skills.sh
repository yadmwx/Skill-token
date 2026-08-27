#!/usr/bin/env bash
# 将 anthropics/skills 仓库中的技能安装到 Cursor 的 .cursor/skills/anthropics-skills/
# 用法: ./scripts/install_anthropics_skills.sh [克隆目录]
# 默认克隆到项目外的 ~/anthropics-skills；也可传入路径，如 .cursor/anthropics-skills-repo

set -e
REPO_URL="https://github.com/anthropics/skills.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CURSOR_SKILLS="${PROJECT_ROOT}/.cursor/skills"
TARGET_DIR="${CURSOR_SKILLS}/anthropics-skills"
CLONE_DIR="${1:-$HOME/anthropics-skills}"

echo "==> Anthropic Skills 安装到 Cursor"
echo "    克隆目录: $CLONE_DIR"
echo "    目标目录: $TARGET_DIR"
echo ""

if [[ ! -d "$CLONE_DIR/.git" ]]; then
  echo "==> 克隆 anthropics/skills ..."
  git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
else
  echo "==> 已存在克隆，拉取最新..."
  (cd "$CLONE_DIR" && git fetch --depth 1 origin main && git checkout main)
fi

SRC_SKILLS="${CLONE_DIR}/skills"
if [[ ! -d "$SRC_SKILLS" ]]; then
  echo "错误: 未找到 $SRC_SKILLS"
  exit 1
fi

mkdir -p "$TARGET_DIR"
echo "==> 创建软链接到 $TARGET_DIR ..."
for d in "$SRC_SKILLS"/*/; do
  name="$(basename "$d")"
  if [[ -d "$d" && -f "$d/SKILL.md" ]]; then
    if [[ -L "$TARGET_DIR/$name" || -d "$TARGET_DIR/$name" ]]; then
      echo "  跳过(已存在): $name"
    else
      ln -sf "$d" "$TARGET_DIR/$name"
      echo "  已链接: $name"
    fi
  fi
done
echo ""
echo "==> 完成。请重启 Cursor 或重新打开项目以加载新技能。"
