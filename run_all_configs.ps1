# 并行运行多个训练配置
# 使用方法: .\run_all_configs.ps1

$configs = @(
    "train/train_GNN_DQN.yaml",
    "train/train_GNN_DQN2hop.yaml",
    "train/train_GNN_PPO.yaml",
    "train/train_NewDDQN_dueling.yaml"
)

$jobs = @()

foreach ($config in $configs) {
    Write-Host "Starting: $config" -ForegroundColor Green
    $job = Start-Job -ScriptBlock {
        param($configPath, $workDir)
        Set-Location $workDir
        python PRC_GNN.py --config $configPath 2>&1
    } -ArgumentList $config, $PWD
    $jobs += $job
}

Write-Host "`nAll $($configs.Count) training jobs started!" -ForegroundColor Cyan
Write-Host "Use 'Get-Job' to check status, 'Receive-Job -Id <id>' to view output" -ForegroundColor Yellow

# 等待所有任务完成（可选）
# Wait-Job -Job $jobs
# 
# 获取所有输出
# foreach ($job in $jobs) {
#     Receive-Job -Job $job
# }
