# A model policy pointer

本实现固定使用 InsightFace `buffalo_l`。权威模型清单与策略分别位于：

- `../manifests/MODEL_MANIFEST.csv`
- `../manifests/MODEL_POLICY.md`

实现中的 `identity_search/model.py` 固化同一组 5 文件大小与 SHA-256，并在构造 `FaceAnalysis` 前离线核验 `<model-root>/models/buffalo_l` 的精确常规文件清单。任何缺失、清单外文件、大小或哈希不符都会停止并生成错误 JSON；搜索调用没有下载分支。
