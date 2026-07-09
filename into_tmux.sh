#!/usr/bin/env bash

# 获取当前所有 tmux 会话名
mapfile -t sessions < <(tmux list-sessions -F "#{session_name}" 2>/dev/null)

# 判断是否存在 tmux 会话
if [ ${#sessions[@]} -eq 0 ]; then
    echo "当前没有 tmux 会话。"
    exit 0
fi

echo "当前 tmux 会话如下："
echo

# 按 1,2,3... 罗列
for i in "${!sessions[@]}"; do
    printf "%d) %s\n" "$((i + 1))" "${sessions[$i]}"
done

echo
read -rp "请输入要进入的会话编号: " choice

# 判断输入是否为数字
if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
    echo "输入无效：请输入数字。"
    exit 1
fi

index=$((choice - 1))

# 判断编号是否越界
if [ "$index" -lt 0 ] || [ "$index" -ge "${#sessions[@]}" ]; then
    echo "输入无效：编号不存在。"
    exit 1
fi

target_session="${sessions[$index]}"

echo "正在进入 tmux 会话：$target_session"

# 如果当前已经在 tmux 内部，则切换会话
# 如果不在 tmux 内部，则 attach 进入会话
if [ -n "$TMUX" ]; then
    tmux switch-client -t "$target_session"
else
    tmux attach-session -t "$target_session"
fi
