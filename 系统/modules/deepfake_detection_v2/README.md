# 二代鉴伪模块（deepfake_detection_v2）

v0.5.0 新增。对 v0.4.0 冻结鉴伪运行时**零改动**，以独立进程 + 独立便携运行时并行提供。

## 分支构成

| 分支 | 内核 | 运行时 | 生效条件 |
| --- | --- | --- | --- |
| `vit_full_image` | CommunityForensics DeepfakeDet-ViT（ViT-Small/384，ONNX FP32） | `packaging_v0.4/resources/portable_python_v2`（onnxruntime 1.30） | 输入短边 ≥ 256px |
| `v040_face_crop` | v0.4.0 MobileNetV3 + 87 维频域统计 | `packaging_v0.3/resources/deepfake_runtime`（冻结） | 输入短边 < 256px，由本模块内部代理 |

路由理由见 `packaging_v0.4/resources/detectors_v2/routing.json`：两个分支的有效域互不重叠，
任何跨域全局加权融合都会把外部集成绩拉低（实测 0.898 → 0.439）。

## 文件

- `persistent_worker_v2.py` —— 协议服务端，协议名与冻结 worker 一致（`face-source-trace-runtime-v1`），
  是鉴伪模块的唯一入口，并按输入短边分流。
- `vit_detector.py` —— ONNX 推理与 CLIP 预处理；不写死任何阈值，分带边界全部读 `calibration.json`。
- `frozen_branch.py` —— 低于阈值时把请求代理给冻结 worker 的常驻子进程封装。

## 运行

```powershell
& .\packaging_v0.4\resources\portable_python_v2\python.exe -E -S -B `
  .\modules\deepfake_detection_v2\persistent_worker_v2.py `
  --writable-root .\runtime\persistent_services\deepfake_v2
```

协议请求与冻结版一致：

```json
{"protocol":"face-source-trace-runtime-v1","type":"request","request_id":"r1",
 "action":"run","query_path":"...","output_path":"..."}
```

## 输出约定

结果仍为 schema 3.0，`details` 中新增：

- `routed_branch` —— 实际生效分支；
- `band` / `band_label` —— `low` 证据不足、`medium` 需人工复核、`high` 高风险提示；
- `band_thresholds` —— 分带边界，来自独立校准集真人图分数的 p95 / p99；
- `short_edge_px` / `routing_threshold_px` —— 路由依据。

## 已知限制

1. 校准样本量小（真人 40 张、换脸 10 张），分带边界应随数据积累重新标定。
2. 路由阈值 256px 由当前两个可用域的输入分布确定，尚无第三个域验证。
3. 本模块不输出"综合可信度"，三个模块的分数依旧不可相加。
