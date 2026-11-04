# 三模块 Streamlit 前端

前端以双搜索 official Streamlit UI 为基线，正式接入 AI 换脸鉴伪，产品标题为“AI换脸涉案图像智能鉴伪与原始素材溯源系统”。

## 交互与真实运行

`Live Analysis` 只有一个涉案图片上传入口，对应四个操作：

- `AI换脸鉴伪`：仅选择 `deepfake`；
- `人脸身份搜索`：仅选择 `identity`；
- `原始底图搜索`：仅选择 `source`；
- `一键综合分析`：选择三个模块。

前端调用 `orchestrator.run_task(..., selected_modules=..., services=...)`，未选模块由后端以 `status=skipped` 记录。一个模块失败不会抹去其他模块结果。

顶部和侧栏使用 `system_status.inspect_system_status()` 的真实检查结果，显示 AIGC 模型、身份数据库、底图数据库和运算设备。显式空库、未初始化或不可用会立即禁用相应按钮，不进入无期限 loading。

`RuntimeServiceManager` 通过 `@st.cache_resource` 在 UI 进程生命周期内启动一次，供 AIGC/身份模型复用。任一服务未就绪时，页面显示降级原因，调度器自动改用隔离子进程，UI 不因服务启动失败而崩溃。

## 结果语义

结果区固定保留三个页签：AI 换脸鉴伪、人脸身份搜索、原始底图搜索。上方的“证据汇总”只并列展示三类证据，绝不计算跨任务融合概率。

AIGC 原生 `LOW_CONFIDENCE` 和 `INSUFFICIENT_EVIDENCE` 保持“不确定”；`is_deepfake=null` 绝不会被映射成 REAL。`score` 只按原模块的 FAKE 风险分数语义显示，不宣称是系统准确率或司法结论。

## Task Result、Mock Demo 与路径安全

- `Task Result` 读取任务根下固定命名的 `detection_result.json`、`identity_result.json` 和 `source_result.json`。
- 旧双模块 task 如缺少 `detection_result.json`，会合成明确的 AIGC `skipped` 占位，不伪造结果。
- `Mock Demo` 只验证三页签、汇总布局和路径解析；其 SVG/JSON 不是模型或准确率证据。
- 资源路径仅允许位于 task、显式批准的库根或 sample 根；拒绝 `..`、越界绝对路径、驱动器相对路径和越界符号链接。
- official 的“关闭系统”保留，并会先幂等关闭持久服务。

完整 JSON 契约见 [TASK_DIRECTORY_CONTRACT.md](TASK_DIRECTORY_CONTRACT.md)。

## 启动与轻量验证

```powershell
python -m streamlit run app.py --server.fileWatcherType none --browser.gatherUsageStats false
```

不启动真实模型的前端单元测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m unittest discover -s tests -v
```
