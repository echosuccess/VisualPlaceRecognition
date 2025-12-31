# 📦 Colab 下载 Logs 完整指南

## 🚀 快速下载（推荐）

### 方法 1：使用自动化脚本

在 Colab 中运行：

```python
!python colab_download_logs.py
```

**这个脚本会：**
1. ✅ 自动打包整个 `logs` 文件夹为 zip 文件
2. ✅ 带时间戳命名（如 `vpr_logs_20231231_143025.zip`）
3. ✅ 显示文件大小和统计信息
4. ✅ 自动触发浏览器下载

---

### 方法 2：直接在单元格中运行

复制以下代码到 Colab 新建单元格：

```python
# === 下载所有 logs 到本地 ===
import shutil
import os
from datetime import datetime
from google.colab import files

# 1. 打包logs
print("📦 正在打包 logs 文件夹...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"vpr_logs_{timestamp}"

# 使用shutil.make_archive创建zip（兼容Drive）
archive_path = shutil.make_archive(
    base_name=zip_name,
    format='zip',
    root_dir='.',
    base_dir='logs'
)

# 2. 显示信息
size_mb = os.path.getsize(archive_path) / (1024 * 1024)
print(f"✅ 打包完成: {archive_path}")
print(f"📊 文件大小: {size_mb:.2f} MB")

# 3. 下载到本地
print("⬇️ 开始下载...")
files.download(archive_path)
print("✅ 完成！文件已保存到浏览器下载文件夹")

# 4. 可选：删除临时zip文件（节省Drive空间）
# os.remove(archive_path)
# print(f"🗑️ 已删除临时文件: {archive_path}")
```

---

## 📋 分步骤说明

### 步骤 1：切换到项目目录

```python
import os
os.chdir('/content/drive/MyDrive/Visual-Place-Recognition-Project')
!pwd
```

### 步骤 2：检查 logs 文件夹

```python
# 查看logs内容
!ls -lh logs/logs/baseline/ | head -20

# 统计实验数量
!ls logs/logs/baseline/ | wc -l
```

### 步骤 3：打包并下载

**选项 A - 完整打包**（推荐）：
```python
!python colab_download_logs.py
```

**选项 B - 只下载特定实验**：
```python
# 例如：只下载 CosPlace 的结果
import shutil
from google.colab import files

shutil.make_archive('cosplace_logs', 'zip', 'logs/logs/baseline', 
                    base_dir='.')
files.download('cosplace_logs.zip')
```

---

## 🛠️ 高级选项

### 只下载结果文件（不含图片）

如果只需要数值结果，不需要可视化图片：

```python
import shutil
import os
from pathlib import Path
from google.colab import files

# 创建临时目录
temp_dir = Path("logs_results_only")
temp_dir.mkdir(exist_ok=True)

# 只复制 results.txt 和 config.yaml
baseline_dir = Path("logs/logs/baseline")
for exp_dir in baseline_dir.iterdir():
    if exp_dir.is_dir():
        exp_name = exp_dir.name
        target_dir = temp_dir / exp_name
        target_dir.mkdir(exist_ok=True)
        
        # 复制结果文件
        for file_name in ["results.txt", "config.yaml", "recall_at_k.txt"]:
            src_file = exp_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, target_dir / file_name)

# 打包并下载
archive = shutil.make_archive("vpr_results_only", "zip", ".", "logs_results_only")
print(f"📦 文件大小: {os.path.getsize(archive) / (1024*1024):.2f} MB")
files.download(archive)

# 清理
shutil.rmtree(temp_dir)
os.remove(archive)
print("✅ 完成！")
```

---

## 📊 下载前预览

查看将要下载的内容：

```python
# 统计信息
!du -sh logs/
!find logs -name "results.txt" | wc -l
!find logs -name "*.jpg" | wc -l

# 预览部分结果
!head logs/logs/baseline/*/results.txt
```

---

## ⚠️ 常见问题

### Q1: 下载很慢或失败？

**原因**：文件太大（包含大量可视化图片）

**解决方案**：
1. 使用"只下载结果文件"方法（见上方高级选项）
2. 分批下载不同方法的结果
3. 直接在 Drive 网页界面下载（右键 logs 文件夹 → 下载）

### Q2: 提示 "No module named 'google.colab'"？

**原因**：不在 Colab 环境中运行

**解决方案**：删除 `files.download()` 这行，文件会保存在当前目录

### Q3: 想要保留 zip 文件在 Drive 上？

```python
# 在打包后不要立即下载，而是移动到指定位置
import shutil
shutil.move(archive_path, f'/content/drive/MyDrive/VPR_Backups/{archive_path}')
print(f"✅ 已保存到: /content/drive/MyDrive/VPR_Backups/")
```

---

## 🎯 推荐工作流

1. **首次下载**：使用 `colab_download_logs.py` 完整下载
2. **后续更新**：只下载新完成的实验结果
3. **备份策略**：定期将 zip 文件保存到 Drive 的备份文件夹

---

## 💾 估计文件大小

根据你的实验配置：
- 每个实验约 5-50 MB（取决于可视化图片数量）
- 当前设置：每个实验保存 3 个 query × 20 张图片
- 预计总大小：**50-200 MB**（视完成的实验数量）

---

## 📝 下载后的建议

1. **解压文件**到本地工作目录
2. **运行分析脚本**：
   ```bash
   python analyze_results.py
   ```
3. **备份重要结果**到云盘或Git仓库
4. **可选**：删除 Colab 上的 logs 以节省 Drive 空间
   ```python
   # ⚠️ 确认已下载备份后再执行！
   # !rm -rf logs/logs/baseline/*
   ```

---

需要帮助？运行诊断：
```python
!python -c "
import os
from pathlib import Path

logs = Path('logs/logs/baseline')
if logs.exists():
    experiments = [d for d in logs.iterdir() if d.is_dir()]
    print(f'实验数量: {len(experiments)}')
    for exp in sorted(experiments)[:5]:
        size = sum(f.stat().st_size for f in exp.rglob('*') if f.is_file())
        print(f'  {exp.name}: {size/(1024*1024):.2f} MB')
    if len(experiments) > 5:
        print(f'  ... 还有 {len(experiments)-5} 个实验')
else:
    print('❌ logs 目录不存在')
"
```
