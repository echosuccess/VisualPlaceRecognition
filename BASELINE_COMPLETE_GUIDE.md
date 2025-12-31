# 📘 Baseline 完整指南

## ✅ 已完成部分

### 1. VPR方法评估（32个实验）

**状态**: ✅ **已完成**

- 评估了4个VPR方法：CosPlace, NetVLAD, MixVPR, MegaLoc
- 测试了2种距离度量：L2 distance vs Dot Product
- 在4个数据集上评估：SF-XS, Tokyo-XS, SVOX-Sun, SVOX-Night
- 总共：4方法 × 2距离 × 4数据集 = 32个实验

**结果**:
- 最佳方法：**MegaLoc** (平均R@1 = 92.62%)
- L2 vs Dot Product：**结果完全相同**（因为描述符已归一化）
- 推荐使用：**Dot Product**（计算更快）

**查看结果**:
```bash
python analyze_results_simple.py
```

**结果文件**:
- `results/baseline_summary.json` - 完整数据
- `results/baseline_table.csv` - CSV表格
- `results/table.tex` - LaTeX表格

---

## ❌ 待完成部分

### 2. Image Matching方法评估 ⭐⭐⭐ **必做**

#### 为什么需要？

这是Baseline的**必做部分**，不是可选项！

项目要求（第5节）明确要求：
> "评估3个Image Matching方法：Superglue, LoFTR, SuperPoint+LightGlue"

#### 目标

1. 对VPR的预测结果运行Image Matching
2. 计算每个query-database配对的**inliers数量**
3. 分析：**inliers是否能作为预测质量的指标？**

#### 实验设计

对于每个VPR方法的结果：
- 读取Top-K预测（如Top-20）
- 对每个query和它的Top-K预测运行Image Matching
- 记录每个配对的inliers数量
- 分析：
  - 正确预测的平均inliers数量
  - 错误预测的平均inliers数量
  - **关键问题**：正确预测是否有更多inliers？

---

## 🚀 运行Image Matching实验

### 准备工作

#### 步骤1：检查环境

```bash
# 检查image-matching-models是否存在
ls image-matching-models/

# 应该看到：
# - matching/
# - main_matcher.py
# - README.md
# - ...
```

✅ 如果文件夹存在，环境已准备好！

#### 步骤2：安装依赖（如果需要）

```bash
cd image-matching-models
pip install -e .

# 如果需要所有方法的依赖
pip install -e .[all]
```

---

### 方法A：快速测试（推荐先尝试）

**目的**：测试环境是否正常，代码是否能运行

```bash
# 只测试MegaLoc + SF-XS + SuperGlue
python run_image_matching_baseline.py \
  --matcher superglue \
  --vpr_method megaloc \
  --dataset sf_xs_test \
  --distance dot_product \
  --top_k 20 \
  --device cuda
```

**预期输出**：
```
================================================================================
Processing VPR experiment: megaloc_dot_product_sf_xs_test
================================================================================
Loaded 100 queries, 200 database images
Will match top-20 predictions per query

Matching queries: 100%|██████████| 100/100 [05:30<00:00,  3.30s/it]

================================================================================
Analysis: Inliers vs Prediction Correctness
================================================================================
Total matches: 2000
Correct predictions: 1712
Incorrect predictions: 288

--- Correct Predictions ---
Mean inliers:   45.23
Median inliers: 42.00
Std inliers:    18.56

--- Incorrect Predictions ---
Mean inliers:   8.34
Median inliers: 5.00
Std inliers:    9.12

--- Difference ---
Mean difference: 36.89
Ratio (correct/incorrect): 5.42x

[CONCLUSION] Correct predictions have significantly MORE inliers!
Inliers can be used as a reliability indicator.
================================================================================
```

**解读**：
- ✅ 如果看到类似输出，说明一切正常！
- ✅ Ratio > 1.5 表示inliers能区分正确/错误预测
- ⏱️ 每个query约3-5秒（取决于GPU）

---

### 方法B：批量运行所有实验

#### 选项1：快速测试模式（推荐初次运行）

```bash
# 只运行MegaLoc + SF-XS的所有3个matcher
python run_all_image_matching.py --quick_test --device cuda
```

**预计时间**：约30-60分钟  
**实验数量**：3个（SuperGlue, LoFTR, SuperPoint-LG）

#### 选项2：运行所有实验（完整）

```bash
# 运行所有VPR方法 × 所有数据集 × 3个matcher
python run_all_image_matching.py --device cuda
```

**预计时间**：约6-12小时  
**实验数量**：约48个（4 VPR × 4 datasets × 3 matchers）

**建议**：
1. 先运行快速测试
2. 确认结果合理后再运行完整版
3. 或者只选择最好的VPR方法（MegaLoc, MixVPR）

---

### 方法C：并行运行（节省时间）

如果你有多个GPU或想在Colab上并行：

**Terminal 1** - 运行SuperGlue:
```bash
python run_all_image_matching.py --device cuda:0
# 修改脚本只运行superglue
```

**Terminal 2** - 运行LoFTR:
```bash
python run_all_image_matching.py --device cuda:1
# 修改脚本只运行loftr
```

---

## 📊 分析结果

### 运行分析脚本

```bash
python analyze_image_matching_results.py
```

**输出**：
1. **控制台**：详细的统计分析
2. **可视化图表**：
   - `results/image_matching/inliers_distribution.png` - 分布对比图
   - `results/image_matching/discrimination_ratio.png` - 区分能力图
3. **LaTeX表格**：
   - `results/image_matching/table.tex` - 用于论文

### 关键指标解读

#### Ratio（区分能力）

```
Ratio = (正确预测的平均inliers) / (错误预测的平均inliers)
```

- **Ratio > 2.0**：🟢 **优秀** - inliers能很好地区分正确/错误预测
- **Ratio 1.5-2.0**：🟡 **良好** - inliers有一定的区分能力
- **Ratio 1.0-1.5**：🟠 **一般** - inliers的区分能力有限
- **Ratio < 1.0**：🔴 **差** - inliers无法区分（甚至反向）

#### 期望结果

根据VPR文献，期望：
- SuperGlue和LoFTR应该有较高的ratio (>1.5)
- 正确的匹配应该有更多的inliers
- 这为Extension 6.1提供了理论支持

---

## 📝 撰写Baseline报告

### 需要回答的问题

#### Q1: 各VPR方法的性能如何？

**答案**：
- MegaLoc最好 (92.62% R@1)
- MixVPR第二 (77.23% R@1)
- CosPlace第三 (64.38% R@1)
- NetVLAD最差 (47.85% R@1)

#### Q2: L2 distance vs Dot product哪个更好？

**答案**：
- 两者结果**完全相同**
- 原因：VPR方法输出的描述符已L2归一化
- 对于归一化向量：L2距离 ≈ 余弦相似度
- **推荐**：使用Dot Product（计算更快）

#### Q3: 各Image Matching方法的性能如何？

**需要基于你的实验结果回答**：
- 哪个方法的inliers最多？
- 哪个方法的区分能力最强（ratio最高）？
- 速度对比如何？

#### Q4: Inliers能否作为预测质量的指标？

**需要基于你的实验结果回答**：
- 正确预测的inliers是否显著多于错误预测？
- Ratio是多少？
- 是否足以用于Extension 6.1的自适应re-ranking？

#### Q5: 最具挑战性的数据集是什么？

**答案**：
- SVOX-Night最困难
- 所有方法在夜间场景的性能都显著下降
- NetVLAD在SVOX-Night只有8.5% R@1

---

## 📋 Baseline Checklist

完整的Baseline包括：

- [x] **Task 1**: 评估4个VPR方法 ✅ **已完成**
  - [x] CosPlace (8/8实验)
  - [x] NetVLAD (8/8实验)
  - [x] MixVPR (8/8实验)
  - [x] MegaLoc (8/8实验)

- [x] **Task 2**: 比较L2 vs Dot Product ✅ **已完成**
  - 结论：两者相同，推荐Dot Product

- [ ] **Task 3**: 评估3个Image Matching方法 ⭐ **待完成**
  - [ ] SuperGlue
  - [ ] LoFTR
  - [ ] SuperPoint + LightGlue

- [ ] **Task 4**: 分析inliers与正确性的关系 ⭐ **待完成**
  - [ ] 计算inliers统计
  - [ ] 分析correct vs incorrect的差异
  - [ ] 得出结论

- [ ] **Task 5**: 撰写Baseline报告 ⭐ **待完成**
  - [ ] 整理所有表格和图表
  - [ ] 回答所有关键问题
  - [ ] 编写Methods和Results章节

---

## 🎯 建议的工作流程

### 今天（如果有时间）

1. **30分钟**：运行快速测试
   ```bash
   python run_all_image_matching.py --quick_test
   ```

2. **10分钟**：查看结果
   ```bash
   python analyze_image_matching_results.py
   ```

3. **决定**：
   - 如果结果合理 → 启动完整实验（后台运行，第二天查看）
   - 如果有问题 → 调试并修复

### 明天

1. **查看完整实验结果**
2. **运行分析脚本**
3. **开始写报告**

### 后天

1. **完成Baseline报告**
2. **决定是否做Extension 6.1**

---

## ⚠️ 常见问题

### Q1: Image Matching很慢怎么办？

**A1**: 
- 减少`--top_k`（如从20改为10）
- 只测试最好的VPR方法（MegaLoc）
- 使用更快的matcher（SuperPoint-LG比LoFTR快）

### Q2: 内存不足？

**A2**:
- 减小图像尺寸（默认512，可以改为256）
- 一次只运行一个matcher
- 使用CPU（会很慢）

### Q3: 必须做Extension 6.1吗？

**A3**:
- 查看你的具体项目要求
- 通常Extension是**加分项**，不是必做
- 但Baseline的Image Matching是**必做**

### Q4: 结果不合理怎么办？

**A4**:
- 检查VPR日志是否正确
- 确认z_data.torch文件完整
- 测试单个样本，查看匹配可视化
- 联系我获取帮助

---

## 📞 需要帮助？

如果遇到问题：
1. 查看错误信息
2. 检查日志文件
3. 尝试简化问题（如只测试1个query）
4. 随时问我！

---

## 🎉 完成Baseline后

恭喜！完成Baseline后你将有：

1. ✅ 完整的VPR方法评估
2. ✅ L2 vs Dot Product的对比结论
3. ✅ Image Matching方法评估
4. ✅ Inliers与正确性关系的分析
5. ✅ 完整的实验报告

**这是一个扎实的视觉位置识别研究基础！** 🎓

---

**现在开始运行Image Matching实验吧！** 🚀
