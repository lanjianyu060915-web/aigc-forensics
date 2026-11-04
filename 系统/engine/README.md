# 通用 AIGC 检测引擎

`aigc_engine` 是从团队的视频检测项目（`deepfake-detection-archive`，提交 `1838f15`）裁出来的推理子集。只保留图片和视频推理需要的部分，音频、训练、外部平台提交和数据准备代码留在原项目。

## 和原项目的差别

| 文件 | 来源 | 改动 |
| --- | --- | --- |
| `models/ufd.py`、`clip_adapt.py`、`community_forensics.py`、`clip_community_fusion.py`、`metrics.py` | 原样复制 | 无 |
| `models/core.py` | 原文件的 `HeadConfig`、`FeatureHead` | 去掉音频模型 |
| `models/checkpoint.py` | 原文件 | 只保留本引擎用到的 4 种结构，去掉保存函数 |
| `media.py` | 原文件的视频部分 | 去掉音频；新增容错解码（默认关闭） |
| `media_tools.py` | 原文件 | 支持 Windows 的 `.exe`，默认找 `系统\ffmpeg` |
| `scorer.py` | 新写 | 单文件打分、图片输入、保留逐帧逐裁剪分数 |

包名从 `deepfake_detection_archive` 改成了 `aigc_engine`。权重文件的格式和哈希不变。

在 4 条真实视频上和原项目的 `predict` 结果比对，分数最大差 2.0e-7。

## 用法

```python
from aigc_engine.scorer import GeneralAigcScorer

scorer = GeneralAigcScorer(
    "models/video-c38-refit3-cf384/weights/video-c38-refit3-cf384.pt",
    sha256="cb13d9c28d404aeec7b1f40208f0731034470d8847bfba55e5cbfe5e67fc3c0f",
)
print(scorer.score_video("a.mp4").logit)
print(scorer.score_image("a.jpg").logit)
```

分数是原始 logit，越高越倾向生成。公式：0.8 × [0.5 × CLIP 原头 + 0.5 × 3 个重训头均值] + 0.2 × CommunityForensics。

图片按一帧视频处理：CLIP 分支取 5 个 224 原尺寸裁剪，CommunityForensics 把短边缩到 440 后中心裁剪 384。视频取 10 帧 × 5 裁剪和 4 帧。

## 测试

需要一个同时装了 torch 和 pytest 的环境：

```bash
python -B -m pytest engine/tests -q -p no:cacheprovider
```
