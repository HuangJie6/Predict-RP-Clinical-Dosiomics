# 项目使用说明

公开内容仅为代码和一张剂量组学特征表（155行，93项数值特征及匿名编号），不含标签、临床数据、分组、CT、剂量体数据或MASK。

无需安装依赖即可检查公开表：`python code/run_analysis.py --stage validate-public`。

完整预测结果需获授权的临床、结局、固定分组及对应特征数据。取得私有输入并放入 `code/data_private/` 后，安装依赖，运行 `python code/run_analysis.py --stage all`。不应把无标签的公开表说成可单独重现完整预测结果的数据集。

完整英文方法、输出位置和各阶段命令见README.md。私有输入和生成结果已通过.gitignore排除。
