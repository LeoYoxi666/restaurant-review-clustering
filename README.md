# 餐馆多维度口碑聚类

本项目将餐馆的口味、环境、服务评分与中文评论信号汇总到商家层级，进行可解释的口碑分群。

## 输入

- `餐馆评价数据集/restaurants.csv`：商家主表。
- `餐馆评价数据集/ratings.csv`：逐条评分与评论。

## 输出

- `outputs/restaurants_labeled.csv`：原商家字段加 `类别标签`。
- `outputs/口碑聚类结果.xlsx`：便于 Excel 查看和筛选的同内容工作簿。
- `docs/口碑分类说明.md`：标签定义、分类原因和经营建议。
- `docs/项目报告.md`：数据质量、方法、模型选择、结果与局限。
- `reports/merchant_evidence.csv`：商家级分类依据，仅供分析和审计。
- `reports/cluster_profiles.csv`：聚类中心和类别规模。

## 运行

```powershell
$python = 'C:\Users\WY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python -m src.pipeline
& $python -m unittest discover -s tests -v
```

Excel 交付由 `tools/build_workbook.mjs` 生成。项目协作规则和运行上下文仅保存在本地，不纳入版本库。

