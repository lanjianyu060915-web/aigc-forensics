# 系统目录技术说明

普通使用看上一级目录的 `README.md` 和 `文档/使用说明.md`。整体结构看 `文档/架构说明.md`。这里只记录改代码和跑评测时要知道的事。

## 三个模型环境

| 目录 | 跑什么 | 怎么启动 |
| --- | --- | --- |
| `python_runtime` | 界面、编排、身份、素材、文件标识、裁脸 | 启动器直接用 |
| `packaging_v0.4/resources/portable_python_v2` | 整图 ONNX 检测 | 带 `-E -S -B` 启动，不读用户环境 |
| `engine_runtime` | 通用 AIGC 引擎、人脸分支打分 | `modules/general_aigc` 的常驻 worker |

这三个目录和 `ffmpeg/` 都不进仓库，用 `tools/setup_runtime.ps1` 按锁定版本重建，`tools/check_runtime.py` 校验。

## 模型服务

`runtime_services.py` 管三个常驻 worker（`deepfake`、`identity`、`general_aigc`），协议是一行一个 JSON（`face-source-trace-runtime-v1`）。worker 不可用时，`orchestrator.py` 回退到隔离子进程，结果相同，只是要重新加载模型。

## 通用 AIGC 引擎

- 代码：`engine/aigc_engine`，来源和改动见 `engine/README.md`。
- 模型包：`models/video-c38-refit3-cf384`，含权重、`manifest.json`（SHA-256）、`calibration.json`（分档边界）和许可记录。
- 视频融合：`0.8 × [0.5 × C38 原头 + 0.5 × 3 个重训头均值] + 0.2 × CF384`。分数是原始 logit，高分更倾向生成。
- 图片融合：`0.5 × 3 个重训头均值 + 0.5 × CF384`。C38 原头在图片上基本失效，不参与。公式由 `ScoreResult.formula` 输出，界面和报告只负责显示。
- 视频：10 帧 × 5 个 224 原尺寸裁剪，另取 4 帧给 CF384（短边 440 双线性，中心裁剪 384）。超过 90 秒的视频按时间点取帧。解码严格模式失败时转容错模式。
- 图片：按一帧处理。

## 人脸分支

`modules/face_crops/run_face_crops.py` 在 `python_runtime` 里运行：

- 视频：读 `frame_selection.json` 的 `largest_face_bbox`，从 `key_frames/` 裁，不再检测。
- 图片：InsightFace 检测最大人脸后裁。

裁剪规则：人脸框中心取边长为长边 1.3 倍的方块，超出画面的部分收缩不补边，人脸框短边不足 40 像素不用。裁剪图存在任务目录的 `face_crops/`。

打分在 `engine_runtime`：`GeneralAigcScorer.score_face_crops` 把裁剪图缩放到 384（双线性），只过 CF384，逐张取 logit，多张取均值。

## 分档和总体规则

- 边界来自真实样本分数的 p95 和 p99，按来源等权。工具：`tools/calibrate_general_aigc.py`，写进模型包的 `calibration.json`，分 `video`、`image`、`face_crop` 三节。
- 总体规则在 `decision/aigc_overall.py`，只由档位推导，分数不相加。改规则要同步改 `tests/test_aigc_overall.py` 和 `文档/架构说明.md`。
- 整图分支的边界在 `packaging_v0.4/resources/detectors_v2/communityforensics_vit/calibration.json`，不影响总体档位。

## 不要动的东西

`modules/deepfake_detection/` 的文件哈希受校验，改了启动检查会失败。小图走的冻结裁剪模型由它加载。

## 评测工具

| 工具 | 用途 |
| --- | --- |
| `tools/eval_general_aigc.py` | 按清单跑通用分支，输出逐样本 JSONL |
| `tools/report_eval.py` | 读逐样本分数，按来源报 AUC、EER 和固定边界下的检出与误报 |
| `tools/calibrate_general_aigc.py` | 由真实样本分数算分档边界 |
| `tools/eval_full_image_video.py` | 整图分支的视频级评测 |
| `tools/build_cross_manifests.py` | 生成换脸数据的交叉评测清单 |
| `tools/recalibrate_detector_v2.py` | 重新标定整图分支 |
| `tools/research/` | 人脸分支的试验脚本：裁脸、提特征、可行性分析 |

## 测试

```bash
cd 系统
python_runtime/python.exe -B -m pytest tests frontend/tests -q -p no:cacheprovider
engine_runtime/python.exe -B tools/run_engine_tests.py
python_runtime/python.exe -B -X utf8 tools/self_check.py
```

`tests/test_provenance.py` 需要 `FFPROBE` 环境变量指向 ffprobe 的绝对路径，否则相关用例跳过。

## 配置

`config.json` 里的路径用 `{APP_ROOT}`、`{DATA_ROOT}`、`{CURRENT_PYTHON}` 占位。环境变量 `FSTS_CONFIG_PATH`、`FSTS_DATA_ROOT` 可覆盖配置文件和数据目录。
