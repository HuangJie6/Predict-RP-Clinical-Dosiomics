# 临床–剂量组学交互分析

本项目包含全肺剂量组学特征提取、特征筛选、随机森林建模、SHAP解释及交互分析代码。

## 数据

`features/whole_lung_dosiomic_features.csv`包含155例样本、93项剂量组学特征及匿名编号。公开数据不包含临床变量、RP标签、队列分组、影像、剂量体数据或MASK。

无需安装额外依赖即可检查公开表：

```bash
python code/run_analysis.py --stage validate-public
```

## 运行

使用Python 3.10或更高版本。安装`requirements.txt`中的依赖后，将获授权的私有输入放在`code/data_private/`，运行：

```bash
python code/run_analysis.py --stage all
```

也可分别执行`select`、`models`、`figures`和`supplement`阶段。输入格式、阶段依赖及输出位置见README.md。私有数据与生成结果已通过`.gitignore`排除。

额外数据申请请联系李雪玲：xlli@cmpt.ac.cn。数据访问须符合机构审批和患者隐私要求。
