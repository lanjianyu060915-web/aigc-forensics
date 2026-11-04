# 第三方资产与许可清单

核对日期：**2026-11-04**。本项目的代码、模型权重和数据分别适用各自的上游条款；本仓库不把整包资产重新许可。权重和训练数据仅按上游条款用于研究，研究用途声明不代替所需授权。本文记录来源与条款，不作法律判断，也不表示所有资产的公开再分发条件均已核实。未取得原文或完整授权链的事项明确标为“未核实”。核对记录和实际访问地址见 `文档/第三方资产核对记录.md`。

## 一、随仓库分发的模型和权重

本表按当前系统包及其上游依赖列示，含已融合进团队 checkpoint 的参数；不表示每一份上游原始文件都单独随包提供。团队训练不改变上游代码、基底权重和训练数据各自的条款。固定版本取自本地来源记录；不能确认固定版本或授权范围的地方不以当前网页替代。

| 名称 | 用途 | 上游地址（固定版本） | 许可 | 限制原文要点与核对边界 |
| --- | --- | --- | --- | --- |
| C38：CLIP ViT-L/14 末层微调与分类头，团队自训 | 图片／视频鉴伪；为融合包提供 CLIP 分支 | 团队模型标识 `video-c38-refit3-cf384-20261103`，来源为下列 UFD、CLIP 和 ViF；没有独立公开上游地址。包内登记 SHA-256：`cb13d9c28d404aeec7b1f40208f0731034470d8847bfba55e5cbfe5e67fc3c0f` | 团队权重未见单独许可证；上游及训练数据条款分别适用；包标记 `research_only` | HD-VILA R-UDA §2.1 限制数据及 Results 用于商业产品／服务，§5.5 明确把符合其定义的训练模型列入 Results。C38 还涉及项目收集媒体，其完整权利链未核实；不能只按 CLIP 的 MIT 标记整个 checkpoint。 |
| 三个重训分类头，团队自训 | 使用 C38 特征，与原头及 CF384 融合 | 同一融合包；本地实验标识 `video-multicrop-refit-fusion-20261103`；无独立公开上游地址 | 独立权重许可、项目收集媒体授权均未核实；ViF 部分见表二 | 训练使用项目既有素材和 ViF 子集。本地许可说明明确记录训练媒体权利未核实、部分标签来自上传者声明，不能把融合当作解除这些限制。 |
| UniversalFakeDetect 分类头 | C38 上游检测器来源链；线性鉴伪头 | [固定版本说明](https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/README.md)，文件 `pretrained_weights/fc_weights.pth`；SHA-256 `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847` | [仓库 MIT LICENSE](https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/LICENSE)；未找到单独分类头模型卡／权重授权说明 | MIT 要求保留版权与许可声明，附免责条款；仓库软件许可证不替代训练素材条款，分类头独立授权范围未核实。该头是历史上游，不声称当前 C38 头与其原始参数相同。 |
| OpenAI CLIP ViT-L/14 | C38 视觉编码器基底 | [内容哈希固定的官方权重地址](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)，地址经[官方加载代码](https://raw.githubusercontent.com/openai/CLIP/main/clip/clip.py)核对 | [仓库 MIT LICENSE](https://raw.githubusercontent.com/openai/CLIP/main/LICENSE)；本次未取得独立权重许可证 | MIT 的保留声明要求适用于其覆盖的软件。[模型卡](https://raw.githubusercontent.com/openai/CLIP/main/model-card.md)将研究列为预期用途，将部署用途列为 out-of-scope，并提示任务内测试；这是模型使用说明，不能改写成 MIT 的非商业条款。训练数据未随官方模型发布。 |
| Community Forensics 384：`OwensLab/commfor-model-384` | 融合包中的完整 ViT-S/16 检测骨干和检测头 | [固定模型卡](https://huggingface.co/OwensLab/commfor-model-384/resolve/6076002bf0d9dd37537f965ee2f06f826c333b61/README.md)，revision `6076002bf0d9dd37537f965ee2f06f826c333b61`；原始 `model.safetensors` 登记 SHA-256 `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` | 模型卡 `license: mit`；[官方代码固定 LICENSE](https://raw.githubusercontent.com/JeongsooP/Community-Forensics/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4/LICENSE) 为 MIT | 保留版权、许可和免责声明。数据集另有研究用途及生成器条款说明，不能把模型卡 MIT 当作数据许可。 |
| CommunityForensics-DeepfakeDet-ViT：`buildborderless` 转换版 | 全图检测分支的 ONNX 推理 | [固定模型卡](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/ac6ee457bea904a373065754107451793b56db00/README.md)，revision `ac6ee457bea904a373065754107451793b56db00`；`onnx/model.onnx` 登记 SHA-256 `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1` | 固定模型卡声明 MIT | 与官方 safetensors 路线分别登记，不能混用 revision、哈希或预处理。模型转换不会替换上游训练数据条款。 |
| InsightFace `buffalo_l` | 人脸检测、对齐与身份特征检索 | [v0.7 模型包发布页](https://github.com/deepinsight/insightface/releases/tag/v0.7)；[v0.7 模型表及许可政策](https://raw.githubusercontent.com/deepinsight/insightface/v0.7/python-package/README.md)。本地固定五个 ONNX 文件，逐文件哈希见 `文档/第三方资产核对记录.md` | **预训练模型仅限非商业研究**；软件代码 MIT 与模型政策分开 | 原文明示手动下载和自动下载的模型都适用非商业研究限制。[当前上游政策](https://raw.githubusercontent.com/deepinsight/insightface/master/README.md)也专门提示 `buffalo_l` 的许可联系渠道。本地文件与 v0.7 发布压缩包的逐字节对应关系未重新核验；不能写成权重 MIT 或无条件开放再分发。 |
| 冻结人脸裁剪分支：MobileNetV3-Small＋87 维频域统计＋逻辑回归 | 人脸裁剪图的空间／频域融合鉴伪 | 项目冻结 checkpoint；MobileNet [torchvision v0.26.0 实现](https://raw.githubusercontent.com/pytorch/vision/v0.26.0/torchvision/models/mobilenetv3.py)。训练基底文件 `mobilenet_v3_small-047dcff4.pth`，登记 SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f` | torchvision 代码 BSD-3-Clause；scikit-learn 代码 BSD-3-Clause；团队训练权重及训练媒体的完整许可链未核实 | 推理 `weights=None` 且加载完整项目 checkpoint，不额外下载官方权重；但包内训练源码明确使用过 ImageNet 预训练，不等于从零训练。[torchvision 权重说明](https://raw.githubusercontent.com/pytorch/vision/v0.26.0/README.md)要求另看训练数据相关条款，不能把代码 BSD 直接用于所有权重。 |

所有哈希均为本次读取的本地清单或源码登记值；本次不是权重二进制一致性复验，也未重新下载大文件。

## 二、训练和评测用到但不随仓库分发的数据

本表区分团队直接训练／评测、上游模型预训练和已排除来源；没有把所有数据都写成 C38 的直接训练集。数据不随仓库分发，并不自动消除训练结果相关条款。

| 名称 | 用途 | 地址 | 许可 | 限制要点 |
| --- | --- | --- | --- | --- |
| ViF／ViF-CoT-4K | C38 来源链、部分重训头训练、开发与外部诊断 | [数据入口](https://huggingface.co/datasets/JoeLeelyf/ViF-CoT-4K)；本地固定 revision `c9e9b30e47e216f72c50664c43750674a79c0ded`；[作者固定版本声明](https://raw.githubusercontent.com/JoeLeelyf/Skyra/51dc07cb95287a586f69172132e477b67da1bb9e/README.md) | 作者声明 CC BY 4.0，**同时要求遵循 Kinetics-400、Panda-70M、HD-VILA-100M 来源条款** | 不能仅用 CC BY 覆盖底层视频。本地 `ViF-provenance.json` 记录只纳入 Panda 来源、排除 Kinetics 媒体训练；这是项目范围记录，不是上游新增授权。HF 数据入口无独立数据卡，本次以作者固定 README 核对声明。 |
| Panda-70M | ViF 部分视频的来源层 | [固定 LICENSE](https://raw.githubusercontent.com/snap-research/Panda-70M/9afd40bc95a0b9ced71513ef9076661a205a1873/LICENSE)；[固定 README](https://raw.githubusercontent.com/snap-research/Panda-70M/9afd40bc95a0b9ced71513ef9076661a205a1873/README.md) | 自定义非商业研究许可 | 数据和代码限非商业研究、学习、教学、测试或验证；再分发须保留声明。README 另要求视频样本遵循 HD-VILA 条款。不是 MIT／Apache 数据集。 |
| HD-VILA-100M | Panda 视频进一步的来源层 | [固定 R-UDA 原文](https://raw.githubusercontent.com/microsoft/XPretrain/902d083e18e753ada0a52b15139619e18d114ecf/hd-vila-100m/LICENSE) | Research Use of Data Agreement v1.0 | §2.1 限非商业计算研究，并限制数据及 Results 用于商业产品／服务；§2.2 不得以数据使用／再分发换取报酬；§3 规定保留归属信息及使下游受 R-UDA 约束。不是绝对禁止一切再分发；本项目选择不分发训练媒体。 |
| Kinetics／Kinetics-400 | ViF 原始来源之一；本地来源记录排除其媒体训练 | [CVDF 官方托管说明](https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/README.md)；[原发布入口](https://deepmind.google/research/open-source/kinetics/) | 本次未核实原始数据许可全文及逐视频授权 | 已核对其 URL／视频片段索引性质；旧官方入口重定向，不能把下载脚本或 I3D 代码的许可当作视频许可。项目来源记录中的 1,196 对排除项不应写成已获授权训练数据。 |
| GenVidBench | 外部评测／诊断，包含记录中的 Kling 400 子集 | [上游 README](https://github.com/genvidbench/GenVidBench)；[作者数据入口](https://huggingface.co/datasets/jian-0/GenVidBench) | 数据声明 CC BY-NC 4.0；仓库根 LICENSE 为 Apache-2.0，二者不能混写 | 数据限制非商业使用；按 CC 条款保留归属、许可链接并标示修改。作者数据入口当前无可读数据卡，本表依据作者 README 的数据声明；本地子集逐项原视频授权及下载 revision 未核实。 |
| AIGVDBench | 历史训练／开发来源与外部诊断、留出评测 | [官方数据卡](https://huggingface.co/datasets/AIGVDBench/AIGVDBench)；[官方说明](https://github.com/LongMa-2025/AIGVDBench) | 数据卡声明 CC BY 4.0 | CC BY 要求归属与修改说明；仓库说明闭源生成模型视频仅用于测试，不包含在训练中，这是上游划分协议而非另一个许可证。本地逐条授权链及所下载 revision 未核实。 |
| SWAN-DF | CF 全图分支阈值校准及验证；案例 01–10 校准、11–20 验证 | [官方演示页](https://swan-df.github.io/)；[Idiap 的 Zenodo v1 记录](https://zenodo.org/records/8365616) | **受 EULA 管理，严格限非商业研究** | 官方记录明确文件访问受限，并要求获授权签署主体满足申请条件。本次未登录、未申请；完整 EULA、项目既有授权以及演示页样本是否存在不同授权均未核实。网页能播放不等于允许打包再分发。 |
| DFDC | 按项目任务口径列为训练／评测数据来源；冻结分支 `data_formal` 与具体 DFDC 批次的映射未核实 | [Meta 官方数据说明](https://ai.meta.com/datasets/dfdc/)；[官方获取入口](https://dfdc.ai/)；[发布平台规则入口](https://www.kaggle.com/c/deepfake-detection-challenge/rules) | **具体数据许可／访问协议未核实** | 官方页区分 Preview 与 Full，要求引用对应论文，获取需账号配置；本次未取得完整协议，不能猜测为 CC BY、MIT 或“仅研究”。演员同意用于上游制作不等于项目任意再分发已获授权。 |
| Community Forensics 数据集 | CF384 上游预训练；本次未发现团队自行全量重训该数据集的证据 | [官方数据卡](https://huggingface.co/datasets/OwensLab/CommunityForensics) | 数据卡同时写 CC BY 4.0 和 **“for research purposes only”** | 原样保留两项表述，不自行裁定其关系。卡片还要求查看逐生成器元数据中的许可；不能将数据集写成 MIT，也不能省略研究用途说明或底层生成器条款。 |
| ImageNet（补充披露） | 冻结 MobileNetV3 训练基底的上游预训练数据；不是本次直接导入的数据集 | [torchvision 预训练权重政策](https://raw.githubusercontent.com/pytorch/vision/v0.26.0/README.md)；[ImageNet 原始入口](https://image-net.org/download.php) | 原数据条款本次未核实 | 官方入口读取失败；不能因推理不下载 ImageNet 权重，就删除预训练来源披露。 |
| OpenFake v2（ComplexDataLab） | 图片评测集：定图片公式和边界，以及一次盲测；真实侧来自 DOCCI、ImageNet、Pexels | [数据卡](https://huggingface.co/datasets/ComplexDataLab/OpenFake)，固定提交 `3247b942f18d39d1fd476b1077852af033d35d8c` | CC-BY-NC-4.0（汇编许可） | 底层图片授权未逐张核实；专有生成器子集另有服务商条款；清单和脚本在 `文档/评测/图片评测-20261104/` |
| Rapidata 文生视频数据集、4DCodeBench/Dataset-Real-World、OmerXYZ/comgenvid（MSVD 子集）、YouZhe/YoCausal-dataset | 视频盲测集 | [Rapidata](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-seedance-1-pro)、[4DCodeBench](https://huggingface.co/datasets/4DCodeBench/Dataset-Real-World)、[comgenvid](https://huggingface.co/datasets/OmerXYZ/comgenvid)、[YoCausal](https://huggingface.co/datasets/YouZhe/YoCausal-dataset)；固定提交见清单 | 数据卡声明 Apache-2.0 或 CC-BY-4.0，78 条未核实 | 底层视频授权未逐条核实；清单和脚本在 `文档/评测/视频盲测-20261104/` |
| 项目早期收集媒体（补充披露） | C38 后续适配、三个重训头及部分验证集合 | 本地实验记录；未保留可完整对应的公开来源地址 | **未核实** | 存在上传者标签及零散来源；不编造统一数据许可，不计为本表其他公开数据集已经覆盖的授权。 |

[CC BY 4.0 原文](https://creativecommons.org/licenses/by/4.0/legalcode.en)和[CC BY-NC 4.0 原文](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)均已于核对日打开。其授权范围不自动包含所有隐私、肖像、商标或其他第三方权利。

## 三、主要软件依赖

版本来自本地元数据，不是查询“最新版”后替换的结果。**应用环境**为 `系统/python_runtime`；**引擎／GPU 环境**为 `系统/engine_runtime` 与训练环境，二者所列核心包版本一致；**冻结环境**为 `系统/packaging_v0.3/resources/deepfake_runtime`。标注“重建清单”的版本仅说明 `系统/requirements-aigc-frozen.txt` 的要求，不冒充全部现存二进制已被验证。

| 名称 | 版本（按环境） | 许可 | 已核对的上游地址 |
| --- | --- | --- | --- |
| Python／CPython | 应用与训练环境 3.12.10；冻结模型元数据也记录 3.12.10 | PSF License Version 2；保留历史及内嵌组件许可 | [v3.12.10 LICENSE](https://raw.githubusercontent.com/python/cpython/v3.12.10/LICENSE) |
| PyTorch／torch | 引擎／GPU：2.11.0+cu128；冻结：2.11.0 | BSD-3-Clause；打包第三方组件另计 | [v2.11.0 LICENSE](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/LICENSE) |
| torchvision | 引擎／GPU：0.26.0+cu128；冻结：0.26.0+cpu | BSD-3-Clause（代码）；预训练权重另看模型与训练数据条款 | [v0.26.0 LICENSE](https://raw.githubusercontent.com/pytorch/vision/v0.26.0/LICENSE) |
| transformers | 引擎／GPU：5.19.0 | Apache-2.0 | [v5.19.0 LICENSE](https://raw.githubusercontent.com/huggingface/transformers/v5.19.0/LICENSE) |
| timm | 引擎／GPU：1.0.30 | Apache-2.0（代码），不覆盖所有预训练权重 | [v1.0.30 LICENSE](https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.30/LICENSE) |
| numpy | 应用：2.4.4；引擎／GPU：2.5.3；冻结：2.5.1 | 主体 BSD-3-Clause；包元数据另列 0BSD、MIT、Zlib、CC0-1.0，内嵌组件见原文件 | [2.4.4](https://raw.githubusercontent.com/numpy/numpy/v2.4.4/LICENSE.txt)、[2.5.3](https://raw.githubusercontent.com/numpy/numpy/v2.5.3/LICENSE.txt)、[2.5.1](https://raw.githubusercontent.com/numpy/numpy/v2.5.1/LICENSE.txt) |
| Pillow | 应用：12.3.0；引擎／GPU：13.0.0；冻结重建清单：12.3.0 | MIT-CMU，不简写为普通 MIT | [12.3.0](https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/LICENSE)、[13.0.0](https://raw.githubusercontent.com/python-pillow/Pillow/13.0.0/LICENSE) |
| OpenCV／opencv-python | 应用：4.13.0.92；冻结重建清单：4.11.0.86 | OpenCV 核心 Apache-2.0；Python 打包脚本 MIT；wheel 内嵌第三方另计 | [OpenCV 4.13.0](https://raw.githubusercontent.com/opencv/opencv/4.13.0/LICENSE)、[4.11.0](https://raw.githubusercontent.com/opencv/opencv/4.11.0/LICENSE)、[打包项目说明](https://raw.githubusercontent.com/opencv/opencv-python/master/README.md) |
| onnxruntime | 应用：1.26.0 | MIT；第三方组件另计 | [v1.26.0 LICENSE](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.26.0/LICENSE) |
| insightface | 应用：1.0.1 | 代码 MIT；预训练模型限非商业研究 | [官方许可政策](https://raw.githubusercontent.com/deepinsight/insightface/master/README.md)；该政策是当前声明，未核实 1.0.1 对应固定源码提交 |
| streamlit | 应用：1.58.0；冻结：1.59.1 | Apache-2.0 | [1.58.0](https://raw.githubusercontent.com/streamlit/streamlit/1.58.0/LICENSE)、[1.59.1](https://raw.githubusercontent.com/streamlit/streamlit/1.59.1/LICENSE) |
| scikit-learn | 冻结：1.9.0 | BSD-3-Clause | [1.9.0 COPYING](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.0/COPYING) |
| joblib | 冻结：1.5.3 | BSD-3-Clause | [1.5.3 LICENSE](https://raw.githubusercontent.com/joblib/joblib/1.5.3/LICENSE.txt) |
| FFmpeg／ffprobe，gyan.dev essentials | 本机 `ffmpeg -version`：9.0.2-essentials_build-www.gyan.dev | **GPL-3.0-or-later**，不是一概 LGPL | [构建发布页](https://www.gyan.dev/ffmpeg/builds/)、[FFmpeg 许可说明](https://ffmpeg.org/legal.html)、[GPLv3 原文](https://www.gnu.org/licenses/gpl-3.0.txt) |

**依赖许可材料的位置与完整性：**完整依赖的逐包许可应从各环境的 `*.dist-info/METADATA`、`LICENSE*`、`COPYING*`、`NOTICE*` 和 `licenses/` 收集；Python 本体另有 `LICENSE.txt`，FFmpeg 独立构建也须另行保留材料。本表不是完整软件物料清单。实际抽查发现一些裁剪环境的 `dist-info` 只有元数据、缺少许可正文，因此不能声称“现有包已附齐全部许可证”。发布时应保留并补齐对应发行包提供的完整许可与第三方声明，不能以本表替代原文。

**FFmpeg 构建特别说明：**本地输出包含 `--enable-gpl --enable-version3 --enable-static`，`-L` 明示 GPL 第 3 版或任何后续版本。`build-source.json` 记录的是滚动下载地址及压缩包哈希，不是对应源代码交付证明。GPLv3 的二进制传播条款涉及 Corresponding Source 的提供方式；应保留该构建的许可、版权信息、准确源码和构建信息。不能用主页链接或另一版本源码替代对应关系，也不在此推断整个 Python 项目自动适用 GPL。OpenCV wheel 内嵌 FFmpeg 的条款与这个独立 gyan.dev 构建分开处理。

## 四、仓库内的图片

| 位置 | 已核对数量 | 来源与授权记录 |
| --- | --- | --- |
| `系统/packaging_v0.3/resources/data/identity_library` | 9 人、45 张人脸照片 | 项目早期整理，未保留逐张来源和授权记录，仅用于测试和演示；如有权利人提出异议会移除。 |
| `系统/packaging_v0.3/resources/data/source_library` | 203 张 | 项目早期整理，未保留逐张来源和授权记录，仅用于测试和演示；如有权利人提出异议会移除。 |
| `系统/examples` | 6 张 | 项目早期整理，未保留逐张来源和授权记录，仅用于测试和演示；如有权利人提出异议会移除。 |

以上说明如实披露现状，不是逐张图片的授权证明，也不将这些图片纳入项目代码许可证。图像来源不明的事实与后续移除承诺应同时保留。

## 五、使用本仓库权重的提示

C38 和重训头的训练媒体含 HD-VILA 来源，R-UDA §5.5 把符合其定义的训练模型列入 Results，§2.1 限制其用于商业产品或服务，§3 要求下游同样受约束。InsightFace `buffalo_l` 限非商业研究。下载和使用这些权重，需要自行遵守对应条款。

FFmpeg 不随仓库分发，`系统/tools/setup_runtime.ps1` 从上游下载固定版本。
