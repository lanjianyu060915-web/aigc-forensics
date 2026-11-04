# 模型包的来源和许可边界

这个模型包是研究候选，未确认商业使用资格。完整清单见仓库根目录的 `THIRD_PARTY.md`。

- C38：CLIP ViT-L/14 末层微调加分类头，由团队在 ViF 等数据上训练。训练媒体的条款见 `HD-VILA-LICENSE.txt`、`Panda-LICENSE.txt` 和 `ViF-provenance.json`，其中 R-UDA 条款限制把数据和结果用于商业产品或服务。
- Community Forensics 384：HuggingFace `OwensLab/commfor-model-384`，版本 `6076002bf0d9dd37537f965ee2f06f826c333b61`，模型卡声明 MIT。
- 3 个重训分类头：在 C38 的 CLIP 特征上训练，训练媒体的权利未核实，标签来自上传者声明或候选方向。
