# 轻量 runner fixtures

这些脚本只用于验证 orchestrator 的子进程故障隔离和服务协议：

- `failing_identity_runner.py` 写入标准错误结果并返回非零退出码；
- `successful_source_runner.py` 写入一个最小成功结果，并输出执行标记。
- `slow_search_runner.py` 休眠，用于验证受控 timeout；
- `mock_persistent_worker.py` 提供带噪声 stdout 的 JSON 行协议假 worker。

它们不导入 A/B 模型、不解码查询图片、不读取真实图库，也不包含或替代任何搜索算法。
