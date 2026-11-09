# GenD 可选权重

第三版默认使用 `cf384` 人脸分支。GenD 适配代码保留，但默认关闭；本仓库不重复分发它约 1.216 GB 的权重。普通默认运行无需下载本文件。

- 上游：[作者模型页](https://huggingface.co/yermandy/GenD_CLIP_L_14)
- 文件名：`model.safetensors`
- 预期大小：`1215928160` 字节
- 固定 SHA-256：`d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833`
- 保存目录：本说明所在的 `系统/models/gend-face/`

需要研究该可选后端时，先从作者模型页下载该文件，再在此目录打开 PowerShell 验证：

```powershell
$modelHash = (Get-FileHash -LiteralPath '.\model.safetensors' -Algorithm SHA256).Hash.ToLowerInvariant()
if ($modelHash -ne 'd76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833') {
    throw '权重摘要不匹配：停止使用，不要修改 manifest 绕过校验。'
}
```

上游文件可能更新；文件名相同不代表内容相同。必须与本目录 `manifest.json` 的固定摘要一致。程序在加载时也会校验模型包；以上步骤不自动启用后端，不训练、校准或修改阈值。

权重许可按上游声明和本目录 `LICENSE.txt` 执行，输入媒体的使用权须另行确认。本仓库的 `.gitignore` 排除此权重，以免再次上传大文件。
