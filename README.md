# Voice-Synced Text Display System

A real-time speech recognition system that displays translated text synchronized with a speaker's voice. Perfect for presentations, conferences, and multilingual events.

## 🎯 Features

- **Real-time Speech Recognition**: Uses OpenAI Whisper for accurate transcription
- **Line-by-Line Synchronization**: Matches spoken words to pre-loaded script lines
- **Keyword Detection**: Highlights specific translated words when keywords are spoken
- **Large Screen Display**: Clean, fullscreen interface optimized for projection
- **Admin Panel**: Easy management of scripts, translations, and keywords
- **Multi-language Support**: Works with any language pair (English ↔ Any Language, etc.)

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│  Speech Recog.  │───▶│   Line Matcher   │
└─────────────────┘    │   (Whisper)     │    └─────────────────┘
                       └─────────────────┘              │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Display       │◀───│   Web Frontend  │◀───│   Flask API     │
│   (Projector)   │    │   (Real-time)   │    │   (Backend)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended for fast, modern Python dependency & virtualenv management)
- Microphone access
- Modern web browser
- Internet connection (for initial Whisper model download)

## 🚀 Quick Start (Recommended: uv)

### 1. 安装 uv

#### 方法一：官方安装脚本（推荐）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 方法二：使用 Homebrew
```bash
brew install uv
```
> ⚠️ 如果 Homebrew 安装失败，请尝试官方安装脚本。

#### 方法三：使用 pipx
```bash
pipx install uv
```

#### 方法四：手动下载
访问 [uv releases](https://github.com/astral-sh/uv/releases) 下载对应平台的二进制文件。

#### 验证安装
```bash
uv --version
```

#### 故障排除
如果安装后 `uv` 命令不可用，请重新加载 shell 配置：
```bash
source ~/.zshrc  # 或 source ~/.bashrc
```

### 2. 配置镜像源（中国用户推荐）

为了提升下载速度，建议配置国内镜像源：

#### 方法一：使用环境变量（推荐）
```bash
# 设置默认镜像源
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/

# 或使用阿里云镜像
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/

# 或使用豆瓣镜像
export UV_DEFAULT_INDEX=https://pypi.douban.com/simple/
```

#### 方法二：使用命令行参数
```bash
# 安装依赖时指定镜像源
uv pip install --default-index https://pypi.tuna.tsinghua.edu.cn/simple/

# 或使用阿里云镜像
uv pip install --default-index https://mirrors.aliyun.com/pypi/simple/
```

#### 方法三：永久配置（添加到 shell 配置文件）
将以下内容添加到 `~/.zshrc` 或 `~/.bashrc`：
```bash
# uv 镜像源配置
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/
```

然后重新加载配置：
```bash
source ~/.zshrc  # 或 source ~/.bashrc
```

#### 验证配置
```bash
# 查看环境变量
echo $UV_DEFAULT_INDEX

# 测试安装（会使用配置的镜像源）
uv pip install --dry-run fastapi
```

> 💡 **提示**：
> - uv 会自动继承系统的 pip 配置，如果你已经配置了 pip 镜像，uv 可能已经使用了相同的镜像源
> - 推荐使用环境变量方式，这样所有 uv 命令都会使用配置的镜像源
> - 如果遇到网络问题，可以尝试不同的镜像源

### 3. 使用 uv 创建虚拟环境并安装依赖

```bash
uv venv  # 创建 .venv 虚拟环境（如未存在）
uv sync  # 自动读取 pyproject.toml 并安装所有依赖
```

### 4. 下载Whisper模型（离线使用）

首次运行前，建议下载Whisper模型以实现离线使用。你可以通过两种方式将 Whisper 模型下载到指定文件夹：**命令行（推荐脚本）** 和 **Web 管理界面**。

#### 方法一：命令行下载到自定义目录

你可以使用项目自带的 `download_whisper_models.py` 脚本，指定模型名称和缓存目录。例如：

```bash
# 下载 base 模型到自定义目录
uv run python download_whisper_models.py --model base --cache-dir /Volumes/1T/whisper-models

# 下载所有模型到默认目录
uv run python download_whisper_models.py

# 或直接下载基础模型到默认目录
uv run python -c "import whisper; whisper.load_model('base')"
```

- `--model`：指定要下载的模型名称（如 tiny、base、small、medium、large 等）。
- `--cache-dir`：指定模型保存的目标文件夹（如外置硬盘路径）。
- 如果目标文件夹不存在，会自动创建。

> ⚠️ 注意：自定义目录下会直接保存模型文件（如 base.pt），不会再嵌套 whisper 子目录。

#### 方法二：通过 Web 管理界面下载到自定义目录

1. 打开 [管理后台](http://localhost:8000/admin) → "Models" 标签页。
2. 在 "Model Cache Directory" 输入框中填写你想要保存模型的文件夹路径（如 `/Volumes/1T/whisper-models`），点击 "Set Directory"。
3. 在 "Download Model" 下拉框选择要下载的模型，点击 "Download"。
4. 下载进度会实时显示，下载完成后模型会保存在你指定的目录下。

> 设置缓存目录后，所有模型的下载和加载都会使用该目录。

**示例：下载 small 模型到外置硬盘**

```bash
uv run python download_whisper_models.py --model small --cache-dir /Volumes/1T/whisper-models
```
或在管理后台设置 `/Volumes/1T/whisper-models` 后，选择 small 并点击 Download。

如需批量下载所有模型，也可在命令行循环调用，或在管理后台点击 "Download All Models"。

### 5. 运行应用

#### 开发模式（推荐，支持热重载）
```bash
# 使用开发脚本（热重载）
uv run python dev.py

# 或使用完整启动脚本（热重载）
uv run python run.py
```

#### 生产模式
```bash
# 直接使用 uvicorn（无热重载）
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

### 6. 访问系统

- **Main Display**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin
- **API Docs**: http://localhost:8000/docs

### 7. 开发模式特性

使用 `dev.py` 或 `run.py` 启动时，系统支持**热重载**功能：

- 🔄 **自动重启**: 修改 `app.py` 或模板文件后，服务器会自动重启
- ⚡ **快速开发**: 无需手动重启服务器，修改代码后立即生效
- 📁 **文件监控**: 监控当前目录下的所有 Python 和模板文件变化
- 🛑 **优雅停止**: 按 `Ctrl+C` 停止服务器

> 💡 **开发提示**：
> - 修改 `app.py` 后，服务器会在 1-2 秒内自动重启
> - 修改 `templates/` 目录下的 HTML 文件也会触发重启
> - 控制台会显示重启信息：`INFO: Detected file change in 'app.py'. Reloading...`

> ⚡ 使用 uv，无需手动激活虚拟环境，所有命令都自动在 .venv 下执行。
> 
> ⚡ 依赖管理全部基于 pyproject.toml，无需 requirements.txt。
> 
> 💡 **提示**：
> - `uv sync` 会自动读取 pyproject.toml 并安装所有依赖，比 `uv pip install` 更简单
> - `uv run` 会自动在虚拟环境中运行命令，无需手动激活虚拟环境

---

## 📖 Usage Guide

### For Speakers

1. **Prepare Your Script**:
   - Upload source language script via admin panel
   - Upload corresponding translations in any target language
   - Add keywords for special highlighting

2. **Start Presentation**:
   - Open display in fullscreen mode
   - Click "Start" to begin speech recognition
   - Speak clearly and follow your script

3. **Controls**:
   - **Spacebar**: Start/Stop listening
   - **R key**: Reset line counter
   - **Mouse**: Use on-screen buttons

### For Administrators

1. **Manage Scripts**:
   - Go to Admin Panel → Scripts tab
   - Enter source language script (one line per sentence)
   - Enter corresponding translations in any target language
   - Save and preview

2. **Configure Keywords**:
   - Upload CSV file with keyword mappings
   - Format: `keyword,translation`
   - Keywords trigger special overlays

3. **System Controls**:
   - Start/Stop speech recognition
   - Reset line counter
   - Monitor system status

## 📁 File Structure

```
voice-synced-text-display/
├── app.py                 # Main FastAPI application
├── run.py                 # User-friendly startup script
├── pyproject.toml         # Project metadata & dependencies (uv/PEP 621)
├── README.md             # Comprehensive documentation
├── .gitignore            # Git ignore file
├── templates/
│   ├── display.html      # Main display interface (fullscreen)
│   └── admin.html        # Admin panel interface
└── data/
    ├── script_en.txt     # Sample English script
    ├── script_translated.txt  # Sample Arabic translations
    └── keywords.csv      # Sample keyword mappings
```

## ⚙️ Configuration

### Script Format

**English Script** (`data/script_en.txt`):
```
Hello, welcome to our presentation.
Today we will discuss important topics.
First, let me introduce our team.
```

**Translated Script** (`data/script_translated.txt`):
```
مرحباً، أهلاً وسهلاً بكم في عرضنا التقديمي.
اليوم سنناقش مواضيع مهمة.
أولاً، دعني أقدم لكم فريقنا.
```

### Keywords Format

**CSV File** (`data/keywords.csv`):
```csv
keyword,translation
hello,مرحباً
welcome,أهلاً وسهلاً
important,مهم
thank you,شكراً لك
```

## 🎨 Customization

### Display Styling

Edit `templates/display.html` to customize:
- Font sizes and colors
- Background gradients
- Animation effects
- Layout positioning

### Admin Panel

Edit `templates/admin.html` to modify:
- Interface layout
- Form styling
- Tab organization
- Control buttons

## 🔧 Advanced Features

### 离线使用配置

系统支持完全离线使用，无需网络连接：

#### Whisper模型缓存位置
- **macOS/Linux**: `~/.cache/whisper/`
- **Windows**: `%USERPROFILE%\.cache\whisper\`

#### 模型文件说明
- `base.pt` - 基础模型 (约145MB) - 当前使用
- `small.pt` - 小型模型 (约461MB) - 更高精度
- `medium.pt` - 中型模型 (约1.42GB) - 高精度
- `large.pt` - 大型模型 (约2.87GB) - 最高精度

---

### Speech Recognition Tuning

In `app.py`, adjust these parameters:
```python
# Similarity threshold for line matching
score > 0.6  # Increase for stricter matching

# Audio processing buffer size
len(audio_buffer) > 16000  # Adjust for latency vs accuracy

# Keyword display duration
time.time() - keyword_display_time < 3  # Seconds to show keywords
```

### Performance Optimization

- Use smaller Whisper models for faster processing
- Adjust audio buffer sizes for your hardware
- Consider using GPU acceleration for Whisper

## 🐛 Troubleshooting

### Common Issues

1. **No Audio Input**:
   - Check microphone permissions
   - Verify audio device selection
   - Test with system audio tools

2. **Poor Recognition**:
   - Speak clearly and slowly
   - Reduce background noise
   - Check microphone quality

3. **Line Matching Issues**:
   - Ensure script lines match spoken content
   - Adjust similarity threshold
   - Review script formatting

4. **Display Not Updating**:
   - Check browser console for errors
   - Verify API endpoints are responding
   - Clear browser cache

### Debug Mode

Run with debug logging:
```python
# In app.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

- Run on local network only
- Use HTTPS in production
- Implement authentication for admin panel
- Validate all file uploads

## 📈 Performance Tips

1. **Hardware Requirements**:
   - Good quality microphone
   - Adequate CPU for real-time processing
   - Sufficient RAM (4GB+ recommended)

2. **Network Considerations**:
   - Local network for best performance
   - Low latency connection between devices
   - Consider wired connections for stability

3. **Browser Optimization**:
   - Use modern browsers (Chrome, Firefox, Safari)
   - Disable unnecessary extensions
   - Fullscreen mode for best display

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue on GitHub
4. Contact the development team

---

**Happy Presenting! 🎤📺**