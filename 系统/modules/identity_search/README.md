# A：人脸身份搜索（第 1 轮工程修复）

本目录是 `identity_search_package` 的工作副本。原件保持只读；本实现保留原算法：
InsightFace `buffalo_l`、`1280x1280` 检测、单脸 L2 归一化 embedding、每人多参考图余弦相似度均值、人员级 Top-K。没有换模型、改聚合方法或用 D 数据调参。

## 运行基线

- Windows x64，Python 3.12.10；
- CPU 为 v0.1 首个基线，明确使用 `CPUExecutionProvider`；
- GPU 是可选能力，不是启动或验收前置条件；
- 直接依赖固定在 `requirements.txt`，完整环境冻结在 `../manifests/A_REQUIREMENTS_LOCK.txt`；
- 环境位于 `../envs/A_env`，不应打入复核包。

从 backend 根目录安装：

```powershell
.\envs\A_env\Scripts\python.exe -m pip install -r .\A_identity\requirements.txt
```

## 离线模型准备与预检

`--model-root` 表示 **InsightFace 根目录**，不是 ONNX 文件目录。模型必须位于：

```text
<model-root>\models\buffalo_l\
```

本机已只读核验的根为 `C:\Users\example\.insightface`，模型目录含 5 个 ONNX 文件、总计 341,268,726 bytes。精确大小与 SHA-256 在 `../manifests/MODEL_MANIFEST.csv`，策略在 `../manifests/MODEL_POLICY.md`。

每次正式调用都在构造 `FaceAnalysis` 前检查目录、完整清单、大小和 SHA-256。缺失、损坏、不可读或出现清单外文件时生成规范错误 JSON 并退出；搜索命令绝不下载模型或访问网络。由于 InsightFace 0.7.3 会间接导入 Albumentations，实现还会在导入 InsightFace 前强制设置 `NO_ALBUMENTATIONS_UPDATE=1`，关闭其 PyPI 版本查询。模型本体不得进入源码或复核包。

## CLI

冻结命令继续有效，默认 CPU：

```powershell
python run_identity_search.py --query query.jpg --library identity_library --top-k 5 --output outputs/identity_result.json
```

建议在任务运行区显式给出缓存与模型根：

```powershell
python run_identity_search.py `
  --query "D:\任务\中文查询.jpg" `
  --library "D:\任务\中文身份库" `
  --top-k 5 `
  --output "D:\任务\结果\identity_result.json" `
  --device cpu `
  --cache-dir "D:\任务\cache\A" `
  --model-root "C:\Users\example\.insightface" `
  --offline
```

可选参数：

- `--device cpu|gpu`：默认 CPU；GPU 必须有 `CUDAExecutionProvider`；
- `--ctx`：兼容旧调用，`-1` 为 CPU，非负数为 GPU ID；
- `--cache-dir`：缓存位置；省略时进入系统用户缓存，绝不默认写源码目录；
- `--model-root`：InsightFace 根；默认 `%USERPROFILE%\.insightface` 或 `INSIGHTFACE_ROOT`；
- `--offline`：显式记录离线意图；即使不写该开关也不会联网。

`--help` 只加载标准库，即使 OpenCV/InsightFace/ONNX Runtime 或模型尚未安装也能显示。

## 输入、输出与诊断

- `top-k` 必须为正整数；超过有效身份数时返回全部有效身份；
- query 必须存在、可解码且恰好检测到一张脸；坏图、无人脸、多人脸分别报错；
- library 必须是目录，一级子目录名同时作为 `identity_id` 和 `identity_name`；
- 支持 `.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`（大小写不敏感）；
- 库内无效图会按原因记录在 `IdentitySearcher.diagnostics` 并跳过；没有任何有效身份时整体报错；
- 人员目录与图片均排序；同分身份以 `identity_id` 排序，结果可重复；
- Windows 图像解码使用 `numpy.fromfile + cv2.imdecode`，JSON 使用 UTF-8 与 `ensure_ascii=False`；
- `runtime_seconds` 是 CLI 端到端时间，包含模型预检、模型初始化、索引和查询；
- 正常错误返回非零退出码，仍先创建 output 父目录并写冻结 schema 的错误 JSON。

成功 JSON 保留 `status/query_image/candidates/runtime_seconds/error_message`，候选保留 `rank/identity_id/identity_name/score/reference_image`。`reference_image` 按调用者传入的 library 路径风格拼接。

v0.4.0 在不改变 `buffalo_l`、512 维 L2 归一化、逐图余弦、人员均值和排序规则的前提下，新增任务级身份检索证据：查询框与五点关键点、对齐前后人脸、Top-K 逐参考图分数、跳过原因、最佳参考对比、真实阶段事件，以及带 SHA-256 的任务相对 POSIX 路径 `artifact_manifest`。证据绘制属于旁路能力，失败只记录 warning，不会把已完成的候选检索改为失败。完整字段和真实回归结果见 `docs/upgrade_v0.4.0/身份检索后端中间证据与回归.md`。

## 安全缓存

缓存由一份 UTF-8 JSON 元数据和一份 `allow_pickle=False` 的 NPZ 特征矩阵组成，不读取历史 `.pkl`。指纹包含：有序文件清单、每文件大小/mtime_ns/SHA-256、冻结模型指纹、检测尺寸和 embedding/聚合策略。图库、模型或关键参数变化都会失效；损坏缓存会安全重建。缓存目录不得放在 identity library 内。

## 测试与合法 smoke 素材

单元测试通过注入 extractor 隔离模型，覆盖 CLI、错误 JSON、坏图/无人脸/多人脸、空库、schema、确定性、中文路径、缓存位置/失效/损坏恢复和 CPU provider。

```powershell
.\envs\A_env\Scripts\python.exe -m pytest .\A_identity\tests -q `
  -p no:cacheprovider `
  --basetemp=.\tests_output\A\tmp `
  --junitxml=.\tests_output\A\junit.xml
```

真实 CPU smoke 只允许使用 A/B/C 原有非 D 图片，并先与 `first_week/data` 做 SHA-256 精确重复比较。当前选择与结果见 `../manifests/A_DEV_SAMPLE_PROVENANCE.csv`；D 图片绝不进入模型推理，smoke 也不冒充准确率验收。
