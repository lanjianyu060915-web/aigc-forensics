# 三模块前端集成报告

日期：2026-08-25  
范围：仅 `face_source_trace_three_module_v0.2_dev/frontend/**`

## 完成项

1. 首页标题更新为“AI换脸涉案图像智能鉴伪与原始素材溯源系统”。
2. `Live Analysis` 改为单一上传入口和四个按钮；四个按钮分别传入 `{"deepfake"}`、`{"identity"}`、`{"source"}` 和三模块集合，不会由前端额外运行未选模块。
3. 顶部与侧栏消费 `inspect_system_status()` 的真实状态/计数；空库、未初始化、资源缺失会禁用对应单模块按钮和综合按钮，并直接显示原因。
4. 使用 `@st.cache_resource` 创建并启动 `RuntimeServiceManager`。构造或启动失败会被转为可视化的降级状态；仍将可用 manager 传给调度器，由调度器按模块回退隔离子进程。
5. `st.status` 显示已选模块、AIGC 模型/图像准备、人脸检测/特征提取/库内搜索、底图特征/库内搜索和最终状态。调度器错误、模块错误和服务降级原因都可在 UI 查看。
6. 结果固定为三页签，页签上方增加三类证据并列汇总。没有融合概率计算。`LOW_CONFIDENCE`、`INSUFFICIENT_EVIDENCE` 和任何 `is_deepfake=null` 保持“不确定”，不映射为 REAL。
7. 保留 `Task Result`、`Mock Demo`、批准资源根/路径边界校验和“关闭系统”。旧双模块 task 缺少 `detection_result.json` 时合成明确的内存 `skipped` 占位，不改写原 task。
8. 新增 AIGC Mock 结果，并扩展 loader/view 以验证和显示 AIGC、`skipped` 及不确定语义。

## 2026-08-25 只读状态快照

- 整体：`available`。
- AIGC：安装程序、适配器、融合配置与三份模型/基础权重校验可用。
- 身份库：9 人、45 张人脸图片、43 条 embedding，index=`ready`。
- 底图库：203 张图片，持久特征数为 0，index=`not_persistent`，实际算法按设计在任务中实时扫描，因此不误判为空库。
- 设备：机器硬件 CUDA 可见，但 AIGC 安装环境为 CPU torch，身份模块也配置 CPU，UI 实际显示 `CPU`。

该快照仅是本次验证时状态；UI 每 5 秒缓存一次检查结果，也可点击“重新检测 / 重试服务”立即刷新。

## 轻量验证

本轮按要求没有启动 Streamlit 真实模型或执行真实推理。

```text
python -B -m unittest discover -s tests -v
Ran 9 tests in 2.359s
OK
```

覆盖：AIGC 不确定语义、三模块 `skipped` schema、旧双模块 task 兼容、三模块 Mock、底图实时扫描状态、实际 CPU 设备显示，以及在明确禁用持久服务的测试配置下进行 Live 首页和 Mock 三页签 Streamlit AppTest smoke。

```text
python -B -c "import app"
APP_IMPORT_OK ('deepfake', 'identity', 'source')
```

另对 `app.py`、`data_loader.py`、`result_views.py`、`status_adapter.py` 执行 AST 解析，4/4 通过。

## 修改文件

- `app.py`
- `components/data_loader.py`
- `components/result_views.py`
- `components/status_adapter.py`（新增）
- `mock_results/detection_result.json`（新增）
- `mock_task/detection_result.json`（新增）
- `mock_task/README.md`
- `tests/test_three_module_frontend.py`（新增）
- `tests/disabled_services_config.json`（新增，仅用于不启动真实模型的 UI smoke）
- `README.md`
- `TASK_DIRECTORY_CONTRACT.md`
- `FRONTEND_INTEGRATION_REPORT.md`（本报告）
