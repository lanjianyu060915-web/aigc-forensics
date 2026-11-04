# 前端任务目录契约 v2

每次上传建立独立任务目录：

```text
case_xxx/
├── query.jpg
├── detection_result.json
├── identity_result.json
├── source_result.json
├── unified_result.json
├── task_status.json
├── visualizations/
└── logs/
```

三份结果 JSON 是 task 根的直接子文件、UTF-8 编码，单文件不超过 5 MiB。状态允许 `success`、`error` 和 `skipped`；`skipped` 明确表示本次未选择模块，不是成功或失败结论。

## AI 换脸鉴伪结果

必需字段：

```text
status, query_image, is_deepfake, score, label,
runtime_seconds, details, error_message
```

- `is_deepfake` 是布尔值或 null。
- `score` 是 0..1 的有限数值或 null，只按原模块的 FAKE 风险分数语义显示。
- `label` 是字符串或 null；`details` 必须是对象。
- `status=success` 且 `label=low_confidence/insufficient_evidence`、`is_deepfake=null` 是不确定结果，不是 REAL。
- `status=error` 必须有非空 `error_message`。
- `status=skipped` 必须使 `is_deepfake=null`、`score=null`，`label` 只能为 null 或 `skipped`。

## 人脸身份与原始底图结果

两份搜索结果的顶层字段：

```text
status, query_image, candidates, runtime_seconds, error_message
```

身份候选字段：

```text
rank, identity_id, identity_name, score, reference_image
```

底图候选字段：

```text
rank, file_path, global_score, local_score, final_score, visualization_path
```

- 候选 `rank` 必须按数组顺序连续为 `1..N`。
- `status=error` 必须有非空错误消息及空候选数组。
- `status=skipped` 必须有空候选数组，可额外携带 `details.reason`。
- `runtime_seconds` 是非负有限数值，为模块口径。

## 旧双模块任务

为保持 official 历史 task 可读，如任务目录有合法的 `identity_result.json` 和 `source_result.json`，但没有 `detection_result.json`，加载器会在内存中生成一份明确的 AIGC `skipped` 占位。原任务文件不会被改写。

## 路径安全

- `query_image` 和匹配可视化应位于 task 根。
- 身份参考图、底图可位于操作者明确批准的对应库根。
- 拒绝 `..`、驱动器相对路径、相对路径冒号、越界绝对路径和越界符号链接。
- 路径通过边界校验后若文件被删除，页面显示受控警告而不崩溃。
