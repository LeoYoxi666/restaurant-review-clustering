# 聚类参数说明

`clustering.json` 是流水线的唯一参数入口。修改参数后必须重新运行流水线和测试。

- `chunk_size`：每次读取的评价行数。数值越大通常越快，但内存占用更高。
- `minimum_reviews_for_clustering`：进入聚类的最低评价数。当前为 3，低于该值标记为“样本不足”。
- `rating_prior_weight`：评分平滑强度。当前为 5，相当于在商家评分中加入 5 条全局均值先验。
- `text_prior_weight`：文本比例平滑强度。当前为 8，用于减弱少量评论造成的极端关键词比例。
- `candidate_k`：待比较的聚类数量。最终结果由轮廓系数和最小簇占比共同选择。
- `random_seed`：固定随机种子，确保相同数据和参数得到相同结果。
- `kmeans_restarts`：每个候选聚类数的重启次数，取惯性最小的结果。
- `kmeans_max_iterations`：单次 K-means 的最大迭代次数。
- `silhouette_sample_size`：计算轮廓系数时的抽样商家数，用于控制大数据内存消耗。
- `feature_weights`：各特征在标准化后的权重。评分维度是主信号，评论主题是辅助信号，因此主题权重较低。
