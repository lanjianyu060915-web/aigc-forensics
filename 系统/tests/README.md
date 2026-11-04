# 快速集成测试

这些测试覆盖 task_id 格式与唯一性、缺失 query 的受控错误、三模块选择与
`skipped` 契约、空库快速失败、子进程超时、服务回退、结构化进度、系统状态，
以及持久工作进程的 JSON 行协议。协议测试会验证同一 PID 跨查询复用、
`ready`/`health`/`request_id` 响应、第三方 stdout 噪音过滤、超时和进程退出检测。

在项目根目录运行：

```powershell
D:\刑侦查竞赛\first_week\_round1_work\backend\envs\A_env\Scripts\python.exe -B -m pytest tests -q
```

测试仅使用 `tests/fixtures` 中的轻量 runner/mock worker，不会初始化或运行
Deepfake、InsightFace、ONNX Runtime 等真实模型，也不会扫描 200 张底图库。
