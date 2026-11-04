# 原始底图搜索模块（队友 B）

输入一张疑似换脸图和候选底图库，返回最可能的原始底图 Top-K、规范 JSON、运行日志与匹配可视化。本包可在 CPU 环境中独立运行。

## 目录

```text
source_search_package/
├── run_source_search.py
├── source_search/
├── requirements.txt
├── requirements.lock.txt
├── README.md
├── sample_data/
├── sample_output/
└── tests/
```

## 安装

支持 Windows 10/11、Python 3.10–3.12，无需 CUDA。请完整解压后，在 `source_search_package` 目录中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本轮已验证组合为 CPython 3.12.10、NumPy 2.5.2、opencv-python
4.14.0.94、Pillow 12.3.0、pytest 9.1.1。需要逐版本重建时改用：

```powershell
python -m pip install -r requirements.lock.txt
```

不要单独移动 `run_source_search.py`，它依赖同级的 `source_search/` 包。

## 运行

```powershell
python run_source_search.py --query sample_data/query/query_01_6514545be6d97386_swap_15ceed878d76a5bf.jpg --library sample_data/source_library --top-k 5 --output sample_output/source_result.json
```

- `--query`：待检索的疑似换脸图。
- `--library`：候选原始底图库目录。
- `--top-k`：返回候选数量。
- `--output`：结果 JSON 路径。

成功时退出码为 0；异常时退出码非 0，并生成含固定顶层字段的错误 JSON。
支持 JPG、JPEG、PNG、WEBP 和 BMP，无法解码的候选图片会被跳过。

每个结果的日志写为 `<output_stem>.runtime.log`；v0.4.0 证据可视化写入：

```text
<output_parent>/visualizations/source/<output_stem>_<query_sha256前12位>_*.jpg
```

因此，同一目录中的不同 output JSON 不会覆盖可视化；即使误用同一个 output
JSON 路径连续运行不同 query，旧可视化也会保留（该 JSON 文件本身仍按调用者指定
被新结果替换）。JSON 内保留绝对 `visualization_path` 供旧调用方读取，同时提供
任务目录相对的 `visualization_relative_path`、顶层 `visual_evidence.source` 和
`artifact_manifest`。manifest 中每个实际证据均含相对 POSIX 路径、SHA-256、
字节数、状态和说明。

安全边界：`--output` 若等于 `--library` 或位于图库内部，命令会在创建任何
目录/文件之前拒绝。因为禁止污染图库，此场景只向标准输出返回规范错误 JSON，
不会在被拒绝的 output 路径落盘。其他正式调用错误（例如非法 `top-k`、缺失
query/library、坏 query、空有效图库或依赖缺失）会尽力写入指定 output JSON。

## 算法

1. 检测人脸并生成非人脸区域掩膜，降低换入人脸对检索的干扰。
2. 提取 HSV 直方图、灰度缩略图、边缘及外围场景特征，进行全库粗检索。
3. 对候选执行 SIFT、ratio test 与 RANSAC 几何验证。
4. 融合全局和局部分数，输出 Top-K 与匹配可视化。

## 输出 JSON

顶层固定字段：`status`、`query_image`、`candidates`、`runtime_seconds`、
`error_message`；v0.4.0 追加 `visual_evidence` 和 `artifact_manifest`，旧字段未删除。

每个候选继续包含：`rank`、`file_path`、`global_score`、`local_score`、
`final_score`、`visualization_path`，并追加全局四分量、局部分解、最终融合分解、
真实 SIFT/Lowe/RANSAC 统计和任务内证据路径。Top1 另有查询/候选关键点、全部
Lowe 匹配、RANSAC 内外点以及可用时的单应投影证据。

完整字段契约和真实案例回归结果见：

- `../../docs/upgrade_v0.4.0/底图溯源后端中间证据接口说明.md`
- `../../docs/upgrade_v0.4.0/底图溯源真实回归测试报告.md`

## 示例数据与来源

- `sample_data/source_library/`：50 张候选原始底图。
- `sample_data/query/`：10 张用于检索验证的换脸式合成查询图。
- `sample_data/ground_truth.csv`：查询图与真实底图映射。
- `sample_data/dataset_sources.csv`：每张原图的 Open Images ID、原始网址、落地页、许可证、作者、尺寸与 SHA-256。
- `sample_data/ATTRIBUTION.md`：数据来源和许可说明。

全部原始图来自 Open Images V7 validation，未使用项目共享视频或共享压缩包中的图片。查询图仅是内部算法测试夹具，由同一独立数据集内其他图片的人脸区域合成，不应作为真实案件样本或深度伪造检测数据集使用。使用、展示或再分发图片前，应按 `dataset_sources.csv` 中每张图片的许可证和署名字段逐项核验。

## 验证结果

- 本轮修复前现场基线：原 5 项测试全部通过（18.64 秒）。
- 本轮修复前完整 sample：50 张图库、query_01，Top-1 命中，5/5 可视化存在，
  算法报告耗时 31.402615 秒。
- 本轮修复后：14 项（原 5 项 + 新增 9 项）全部通过；控制台 17.27 秒，
  JUnit suite 17.106 秒。
- 修复后测试、sample 与双 query 的机器可核验证据见
  `../reports/B_REPAIR_REPORT.md`、`../tests_output/B/`、`../runtime/B/` 和
  `../logs/B_*.log`（相对于本轮 backend 工作区）。

原作者 README 曾报告 50 张开发集 10/10 和未交付 150 张留出集 10/10；本轮没有
把这些历史数字当作动态证据，也没有执行 D 正式独立验收。硬件与后台负载会影响耗时。

## 测试

```powershell
python -m pytest -q --basetemp ..\tests_output\B\pytest_temp `
  -o cache_dir=..\tests_output\B\pytest_cache
```

保留原 5 项测试，并覆盖无第三方依赖的 CLI help、依赖失败/非法 Top-K 错误 JSON、
output/library containment、完整冻结 schema、小图库与 K 超界、坏候选、无人脸退化、
多人脸取最大脸、中文路径，以及跨 query 可视化永久不覆盖。

## 已知限制

- 当前检索封闭图片库，不支持视频片段或互联网全网反向搜索。
- 多人脸时默认处理面积最大的人脸；未检出人脸时退化为全图检索。
- 强裁剪、大面积背景替换、严重模糊或极低分辨率可能降低召回率。
- 分数只表示候选库内的相对相似性，不构成司法来源认定；实际案件中还需结合原文件哈希、传播链及人工复核。
