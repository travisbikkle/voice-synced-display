# 🎤 Whisper模型管理功能

## 📋 功能概述

现在系统支持完整的Whisper模型管理功能，包括：

- ✅ **模型选择** - 在管理界面选择不同的Whisper模型
- ✅ **模型下载** - 一键下载所需的模型
- ✅ **自定义缓存目录** - 将模型存储到指定位置
- ✅ **离线使用** - 下载后完全离线运行
- ✅ **状态监控** - 实时查看模型下载状态

## 🎯 新增功能

### 1. 管理界面新增"Models"标签页

在管理界面 (`http://localhost:8000/admin`) 新增了"Models"标签页，包含：

#### 模型选择区域
- **当前模型下拉框** - 显示所有可用模型，已下载的显示✅标记
- **缓存目录设置** - 可以指定自定义的模型存储位置

#### 模型下载区域
- **单个模型下载** - 选择特定模型进行下载
- **批量下载按钮** - 一键下载基础模型或所有模型

#### 状态显示区域
- **当前模型信息** - 显示正在使用的模型
- **缓存目录信息** - 显示模型存储位置
- **模型列表** - 显示所有模型的下载状态

### 2. 新增API端点

#### `GET /api/whisper-models`
获取所有可用模型及其状态
```json
{
  "models": [
    {
      "name": "base",
      "downloaded": true,
      "size_mb": 138.5,
      "current": true
    }
  ],
  "current_model": "base",
  "cache_dir": "/Users/yu/.cache/whisper"
}
```

#### `POST /api/set-model`
设置要使用的模型
```json
{
  "model_name": "small"
}
```

#### `POST /api/set-model-cache`
设置模型缓存目录
```json
{
  "cache_dir": "/custom/path/to/models"
}
```

#### `POST /api/download-model`
下载指定模型
```json
{
  "model_name": "medium"
}
```

## 🚀 使用方法

### 1. 访问管理界面
```
http://localhost:8000/admin
```

### 2. 切换到"Models"标签页
点击导航栏中的"Models"标签

### 3. 选择模型
- 在"Current Model"下拉框中选择要使用的模型
- 已下载的模型会显示✅标记和文件大小
- 当前使用的模型会显示"(Current)"标记

### 4. 下载模型
- **单个下载**: 在"Download Model"下拉框选择模型，点击"Download"
- **批量下载**: 点击"Download Basic Models"或"Download All Models"

### 5. 设置缓存目录（可选）
- 在"Model Cache Directory"输入框中输入自定义路径
- 点击"Set Directory"保存设置

## 📁 模型存储位置

### 默认位置
- **macOS/Linux**: `~/.cache/whisper/`
- **Windows**: `%USERPROFILE%\.cache\whisper\`

### 自定义位置
可以通过管理界面或环境变量设置：
```bash
export XDG_CACHE_HOME=/your/custom/path
```

## 📊 可用模型对比

| 模型 | 大小 | 精度 | 速度 | 推荐用途 |
|------|------|------|------|----------|
| `tiny` | 72MB | 低 | 最快 | 快速测试 |
| `base` | 145MB | 中 | 快 | 日常使用 ✅ |
| `small` | 461MB | 高 | 中等 | 高质量识别 |
| `medium` | 1.42GB | 很高 | 慢 | 专业用途 |
| `large` | 2.87GB | 最高 | 最慢 | 研究用途 |

## 🔧 高级功能

### 1. 命令行下载工具
```bash
# 下载特定模型
uv run python download_whisper_models.py --model base

# 下载到自定义目录
uv run python download_whisper_models.py --model small --cache-dir /custom/path

# 列出所有模型
uv run python download_whisper_models.py --list

# 静默模式
uv run python download_whisper_models.py --model tiny --quiet
```

### 2. 离线使用配置
1. 下载所需模型
2. 断开网络连接
3. 正常使用系统

### 3. 模型切换
- 在管理界面选择新模型
- 系统会自动重新加载模型
- 无需重启应用

## 🧪 测试功能

运行测试脚本验证功能：
```bash
uv run python test_models.py
```

测试包括：
- ✅ 获取模型列表
- ✅ 设置模型
- ✅ 设置缓存目录
- ✅ 下载模型

## 💡 使用建议

### 日常使用
- 推荐使用 `base` 模型，平衡了精度和速度
- 下载 `tiny`, `base`, `small` 三个基础模型

### 高质量需求
- 使用 `small` 或 `medium` 模型
- 确保有足够的存储空间

### 离线环境
- 提前下载所需模型
- 设置合适的缓存目录
- 测试离线功能

### 存储优化
- 将模型存储在SSD上以提高加载速度
- 定期清理不需要的模型
- 使用自定义缓存目录便于管理

## 🔍 故障排除

### 模型下载失败
1. 检查网络连接
2. 确认有足够的磁盘空间
3. 检查缓存目录权限

### 模型加载失败
1. 确认模型已下载完成
2. 检查缓存目录路径
3. 重启应用

### 权限问题
1. 确保缓存目录可写
2. 检查应用运行权限
3. 使用绝对路径

## 📈 性能优化

### 模型选择
- 根据硬件配置选择合适的模型
- 考虑实时性要求

### 存储优化
- 使用SSD存储模型文件
- 定期清理旧版本模型

### 内存管理
- 大型模型需要更多内存
- 监控系统资源使用

---

🎉 **现在你可以完全控制Whisper模型，实现真正的离线语音识别！** 