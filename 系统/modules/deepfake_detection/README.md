# deepfake_detection 冻结模型适配与证据层

本目录调用应用自带 `deepfake_runtime` 中的正式模型，不重训模型，也不修改预处理、权重、阈值或最终评分算法。

- 空间域：`mobilenet_v3_small(weights=None)` + 项目完整二分类 checkpoint（`strict=True`）
- 频域：87 维真实统计特征 + `StandardScaler` + `LogisticRegression`
- 融合：权重和阈值始终从 `models/fusion_formal/fusion_config.json` 动态读取
- 设备：正式包使用 CPU 运行时；设备状态与模型状态分开报告
- 官方 ImageNet MobileNet 权重：正式推理不需要，也不访问用户 Torch 缓存

## 调用方式

```powershell
python.exe -E -S -B run_deepfake_detection.py `
  --query "<查询图片>" `
  --output "<任务目录>\detection_result.json" `
  --install-root "<应用目录>\deepfake_runtime" `
  --device cpu
```

持久服务 `persistent_worker.py` 使用同一映射和证据生成函数，首次加载模型后可串行复用，不会改变结果契约。

## 结果语义

`score` 是空间域和频域按当前配置融合后的 FAKE 风险筛查分数，不是系统准确率或司法鉴定结论。`label=deepfake`/`real` 只映射正式模型的阈值比较结果；无人脸返回 `label=no_face`、空分数和明确状态，不伪造真假结论。

旧字段保持不变。新增顶层字段：

- `visual_evidence.deepfake`：本次任务的结构化可视化证据
- `artifact_manifest`：全部 AIGC 工件的任务相对 POSIX 路径、SHA-256 和大小
- `progress_events`：模型管线在真实代码执行点记录的阶段、起止时间、耗时和真实 payload

### `visual_evidence.deepfake` schema

- `face_detection`
  - 原图实际人脸框、正式扩展裁剪框、选中人脸序号、224×224 模型裁剪
  - 关键点仅登记 OpenCV 眼部级联分类器真实检出的候选中心；若未检出则为空，不用几何估算点冒充真实关键点
- `spatial`
  - `method=occlusion_sensitivity`
  - 对正式 MobileNetV3-Small 的当前 FAKE 输出执行 6×6 批量遮挡前向
  - 返回有符号分数变化、绝对敏感性、纯热图、叠加图、最强响应网格及耗时
  - 该证据只描述模型敏感性，不等于像素级伪造定位
- `frequency`
  - 实际模型人脸裁剪的未中心化/中心化二维 FFT 对数幅度谱
  - 32 段径向频谱数据和曲线图
  - 对真正送入 LogisticRegression 的 StandardScaler 输出计算“模型输入值 × 回归系数”
  - 返回 87 维完整贡献、绝对贡献 Top-10、正/负贡献 Top-10，并验证重建 FAKE 概率
- `fusion`
  - 空间/频域分数、动态权重、加权分量、融合分数、阈值、差值、公式代入和规范结论
- `generation`
  - 证据生成状态、耗时、工件数量和失败隔离说明

`artifact_manifest` 每项包含：

```json
{
  "id": "deepfake.spatial_overlay",
  "module": "deepfake",
  "kind": "spatial_sensitivity_overlay",
  "path": "visualizations/deepfake/evidence/deepfake_<输入摘要>_spatial_occlusion_overlay.jpg",
  "sha256": "...",
  "size_bytes": 12345,
  "status": "generated",
  "description": "..."
}
```

路径相对于任务目录，统一使用 `/`，兼容中文和空格路径。冻结管线原生 JSON、CSV、HTML、运行日志及关键帧也会登记进同一 manifest。

## 失败隔离

可视化证据属于检测后的解释性计算。任何热图、FFT、贡献恢复或文件写入异常都会被捕获，并返回 `visual_evidence.deepfake.status=unavailable` 和技术原因；已经得到的检测状态、空间/频域/融合分数、阈值和最终结论不会被改写。

## 验证

专项测试见 `tests/test_deepfake_visual_evidence.py`，覆盖：

- 正式 AI 换脸样例真实推理及数值回归；
- 12 项以上证据工件的路径、大小和 SHA-256；
- Occlusion、FFT、径向频谱和 87 维贡献真实性；
- LogisticRegression 概率重建误差；
- 无人脸可视化失败状态；
- 持久工作进程契约；
- 解释失败不影响检测结论。
