# 示例图片

- `aigc_real_image.jpg`、`aigc_fake_image.jpg`、`no_face_image.jpg`：来自已安装 DeepfakeDetectionSystem 的 `samples/demo`，用于合同/错误状态回归。REAL/FAKE 名称来自原交付 manifest；本集成不根据文件名生成模型结论。
- `integrated_query_001.jpg`—`003.jpg`：来自 `first_week/data/换脸查询图`，是既有双搜索三例回归素材；历史期望 Rank-1 分别为身份 `person_001/002/003`、底图 `122.jpg/107.jpg/019.jpg`。

所有示例均只用于测试和演示。AIGC score、身份相似度和底图相似度属于不同任务，不得相加为综合概率。
