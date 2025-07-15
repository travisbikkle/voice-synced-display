from fastapi import FastAPI, Request, File, UploadFile, Form, Body, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import whisper
import sounddevice as sd
import numpy as np
import pandas as pd
import threading
import time
import os
import re
from difflib import SequenceMatcher
import json
from typing import Dict, List, Optional
from pydantic import BaseModel
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
import collections
import torch  # Add this at the top-level imports if not present

print(f"[DEBUG] Loaded app.py from: {__file__}")

app = FastAPI(
    title="Voice-Synced Text Display System",
    description="Real-time speech recognition with synchronized text display",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables for speech recognition
model = None
# 不再需要 opencc_s2t
audio_buffer = []
is_listening = False
current_line_index = 0
current_keyword = None
keyword_display_time = 0
current_transcribed_text = ""  # 当前 Whisper 识别的原始文本
audio_thread = None
stream = None
selected_device_id = None  # 当前选择的音频设备ID
selected_model_name = "base"  # 当前选择的模型名称
model_cache_dir = None  # 自定义模型缓存目录
download_progress = {}  # 下载进度跟踪
active_download_threads = {}  # 记录每个模型的下载线程
MATCH_THRESHOLD = 0.8  # 句子匹配相似度阈值（可通过API动态调整）
RECOGNITION_LANGUAGE = None  # None=auto, 'en', 'zh', ...
RECOGNITION_LANGUAGE_UI = None  # UI选择的语言（如 zh-Hans, zh-Hant, en）
AUDIO_BUFFER_SECONDS = 2.0   # 音频缓冲区长度（秒）
SILENCE_THRESHOLD = 0.01  # 能量阈值，默认
SILENCE_DURATION = 0.5    # 静音判定时长（秒）

# Load scripts and keywords
script_en = []
script_translated = []
keywords_mapping = {}

# Pydantic models for API
class ScriptData(BaseModel):
    english: Optional[List[str]] = None
    translated: Optional[List[str]] = None

class KeywordData(BaseModel):
    keyword: str
    translation: str

class AudioDeviceData(BaseModel):
    device_id: int

class ModelData(BaseModel):
    model_name: str

class ModelCacheData(BaseModel):
    cache_dir: str

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.last_push_data = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        self.last_push_data = data
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

def load_scripts():
    """Load English script and translated script"""
    global script_en, script_translated
    
    try:
        with open('data/script_en.txt', 'r', encoding='utf-8') as f:
            script_en = [line.strip() for line in f.readlines() if line.strip()]
        
        with open('data/script_translated.txt', 'r', encoding='utf-8') as f:
            script_translated = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Loaded {len(script_en)} English lines and {len(script_translated)} translated lines")
    except FileNotFoundError:
        print("Script files not found. Please create data/script_en.txt and data/script_translated.txt")
        # Create sample data in English
        script_en = ["Hello, welcome to our presentation.", "Today we will discuss important topics."]
        script_translated = ["Hello, welcome to our presentation.", "Today we will discuss important topics."]

def load_keywords():
    """Load keyword mapping from CSV file"""
    global keywords_mapping
    
    try:
        df = pd.read_csv('data/keywords.csv')
        keywords_mapping = dict(zip(df['keyword'].str.lower(), df['translation']))
        print(f"Loaded {len(keywords_mapping)} keyword mappings")
    except FileNotFoundError:
        print("Keywords file not found. Creating sample keywords.csv")
        # Create sample keywords in English
        keywords_mapping = {
            'hello': 'Hello',
            'welcome': 'Welcome',
            'important': 'Important',
            'thank you': 'Thank you',
            'goodbye': 'Goodbye'
        }
        # Save sample keywords
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(list(keywords_mapping.items()), columns=['keyword', 'translation'])
        df.to_csv('data/keywords.csv', index=False)

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(transcribed_text, threshold=None):
    """Find the best matching line in the script"""
    global current_line_index
    
    if not script_en:
        return -1
    
    if threshold is None:
        threshold = MATCH_THRESHOLD
    
    best_match = -1
    best_score = 0
    
    # Check current line and nearby lines
    start_idx = max(0, current_line_index - 2)
    end_idx = min(len(script_en), current_line_index + 3)
    
    for i in range(start_idx, end_idx):
        score = similarity(transcribed_text, script_en[i])
        if score > best_score and score >= threshold:  # Threshold for matching
            best_score = score
            best_match = i
    
    return best_match

def detect_keywords(transcribed_text):
    """Detect keywords in transcribed text"""
    global current_keyword, keyword_display_time
    
    text_lower = transcribed_text.lower()
    detected_keywords = []
    
    for keyword, translation in keywords_mapping.items():
        if keyword in text_lower:
            detected_keywords.append((keyword, translation))
    
    if detected_keywords:
        # Use the first detected keyword
        current_keyword = detected_keywords[0]
        keyword_display_time = time.time()
        return current_keyword
    
    return None

def audio_callback(indata, frames, time, status):
    """Callback for audio input"""
    global audio_buffer
    if is_listening:
        audio_buffer.extend(indata[:, 0])

def is_silence(audio_chunk, threshold=None):
    if threshold is None:
        threshold = SILENCE_THRESHOLD
    return np.mean(np.abs(audio_chunk)) < threshold

def process_audio_buffer(buffer):
    global current_line_index, current_keyword, current_transcribed_text, RECOGNITION_LANGUAGE, RECOGNITION_LANGUAGE_UI
    audio_data = np.array(buffer)
    if np.max(np.abs(audio_data)) == 0:
        return
    audio_data = audio_data / np.max(np.abs(audio_data))
    try:
        kwargs = {}
        if RECOGNITION_LANGUAGE_UI in ["en", "ja", "ko", "es", "fr", "de", "ru", "it", "pt", "ar", "hi"]:
            kwargs['language'] = RECOGNITION_LANGUAGE_UI
        elif RECOGNITION_LANGUAGE_UI == "auto" or RECOGNITION_LANGUAGE_UI is None:
            pass
        elif RECOGNITION_LANGUAGE:
            kwargs['language'] = RECOGNITION_LANGUAGE
        result = model.transcribe(audio_data, **kwargs)
        transcribed_text = result["text"].strip()
        if transcribed_text:
            current_transcribed_text = transcribed_text
            matched_line = find_best_match(transcribed_text, threshold=MATCH_THRESHOLD)
            if matched_line != -1:
                current_line_index = matched_line
            keyword = detect_keywords(transcribed_text)
            try:
                global MAIN_LOOP
                if MAIN_LOOP and MAIN_LOOP.is_running():
                    import asyncio
                    asyncio.run_coroutine_threadsafe(manager.broadcast({
                        'line_index': current_line_index,
                        'english_text': script_en[current_line_index] if current_line_index < len(script_en) else '',
                        'translated_text': script_translated[current_line_index] if current_line_index < len(script_translated) else '',
                        'keyword': keyword[0] if keyword else None,
                        'keyword_translation': keyword[1] if keyword else None,
                        'transcribed_text': current_transcribed_text
                    }), MAIN_LOOP)
            except Exception as push_err:
                print(f"WebSocket push error: {push_err}")
    except Exception as e:
        print(f"Error in speech recognition: {e}")

def audio_processing_thread():
    global audio_buffer, is_listening
    silence_buffer = collections.deque(maxlen=int(SILENCE_DURATION * 10))  # 0.1s一帧
    chunk_size = int(0.1 * 16000)  # 0.1秒音频
    temp_buffer = []
    while is_listening:
        if len(audio_buffer) >= chunk_size:
            chunk = audio_buffer[:chunk_size]
            audio_buffer = audio_buffer[chunk_size:]
            temp_buffer.extend(chunk)
            silence_buffer.append(is_silence(np.array(chunk)))
            if len(silence_buffer) == silence_buffer.maxlen and all(silence_buffer):
                if len(temp_buffer) > int(0.5 * 16000):
                    process_audio_buffer(temp_buffer)
                temp_buffer = []
        else:
            time.sleep(0.05)

def start_listening():
    """Start listening for speech input"""
    global is_listening, model, audio_thread, stream, selected_device_id, selected_model_name, model_cache_dir
    
    if model is None:
        print(f"Loading Whisper model: {selected_model_name}")
        try:
            import whisper
            # 只在设置了自定义目录时传递 download_root
            if model_cache_dir:
                print(f"Using custom cache directory: {model_cache_dir}")
                model = whisper.load_model(selected_model_name, download_root=model_cache_dir)
            else:
                print("Using default cache directory (~/.cache/whisper)")
                model = whisper.load_model(selected_model_name)
            print(f"Model {selected_model_name} loaded successfully from cache")
        except Exception as e:
            print(f"Error loading model {selected_model_name} from cache: {e}")
            print("Please ensure the model is downloaded for offline use")
            return
    
    if not is_listening:
        try:
            # Get device ID - use selected device or default device
            device_id = selected_device_id
            devices = sd.query_devices()
            # 新增：如果 id 不在设备列表，尝试用 name 匹配
            if device_id is not None:
                if not (0 <= device_id < len(devices)):
                    # 尝试用 name 匹配
                    try:
                        from app import selected_device_name
                    except ImportError:
                        selected_device_name = None
                    if selected_device_name:
                        for i, d in enumerate(devices):
                            if d['name'] == selected_device_name:
                                device_id = i
                                selected_device_id = i
                                print(f"Fallback to device by name: {selected_device_name} (id={i})")
                                break
                        else:
                            print(f"Device name {selected_device_name} not found, fallback to system default")
                            device_id = sd.default.device[0]
                            selected_device_id = None
                # id 合法
            else:
                # None 表示系统默认
                device_id = sd.default.device[0]
            # Get device info for logging
            if 0 <= device_id < len(devices):
                device_name = devices[device_id]['name']
                print(f"Starting audio stream on device {device_id}: {device_name}")
            else:
                print(f"Warning: Invalid device ID {device_id}, falling back to default")
                device_id = sd.default.device[0]
                device_name = devices[device_id]['name']
                print(f"Using fallback device {device_id}: {device_name}")
            
            stream = sd.InputStream(
                callback=audio_callback, 
                channels=1, 
                samplerate=16000,
                device=device_id
            )
            stream.start()
            print("Audio stream started successfully")
            
            # Set listening flag
            is_listening = True
            
            # Start audio processing thread
            audio_thread = threading.Thread(target=audio_processing_thread)
            audio_thread.daemon = True
            audio_thread.start()
            
            print(f"Started listening for speech on device {device_id} ({device_name})...")
            print(f"is_listening flag set to: {is_listening}")
            
        except Exception as e:
            print(f"Error starting listening: {e}")
            is_listening = False
            if stream:
                stream.stop()
                stream.close()
                stream = None
            raise e

def stop_listening():
    """Stop listening for speech input"""
    global is_listening, stream, audio_thread
    
    is_listening = False
    
    if stream:
        stream.stop()
        stream.close()
        stream = None
    
    if audio_thread:
        audio_thread.join(timeout=1)
        audio_thread = None
    
    print("Stopped listening for speech...")

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main display page"""
    return templates.TemplateResponse("display.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Admin panel"""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/current-line")
async def get_current_line():
    """Get current translated line"""
    global current_line_index
    
    if current_line_index < len(script_translated):
        return {
            'line_index': current_line_index,
            'translated_text': script_translated[current_line_index],
            'english_text': script_en[current_line_index] if current_line_index < len(script_en) else ""
        }
    
    return {'line_index': -1, 'translated_text': '', 'english_text': ''}

@app.get("/api/keyword")
async def get_current_keyword():
    """Get current keyword if any"""
    global current_keyword, keyword_display_time
    
    if current_keyword and time.time() - keyword_display_time < 3:  # Show for 3 seconds
        return {
            'keyword': current_keyword[0],
            'translation': current_keyword[1]
        }
    
    return {'keyword': None, 'translation': None}

@app.get("/api/transcribed-text")
async def get_transcribed_text():
    """Get current transcribed text for debugging"""
    global current_transcribed_text
    return {'transcribed_text': current_transcribed_text}

@app.get("/api/scripts")
async def get_scripts():
    """Get all scripts for admin panel"""
    load_scripts()  # 每次请求都重新读取文件
    return {
        'english': script_en,
        'translated': script_translated,
        'keywords': keywords_mapping
    }

@app.post("/api/upload-script")
async def upload_script(data: ScriptData):
    """Upload new script files"""
    global script_en, script_translated
    
    try:
        if data.english is not None:
            script_en = data.english
            with open('data/script_en.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(script_en))
        
        if data.translated is not None:
            script_translated = data.translated
            with open('data/script_translated.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(script_translated))
        
        return {'success': True, 'message': 'Scripts updated successfully'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.post("/api/upload-keywords")
async def upload_keywords(file: UploadFile = File(...)):
    """Upload new keywords file"""
    global keywords_mapping
    
    try:
        # Read CSV content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Create temporary file to read with pandas
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(content_str)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_csv(tmp_file_path)
            keywords_mapping = dict(zip(df['keyword'].str.lower(), df['translation']))
            
            # Save to file
            df.to_csv(csv_path, index=False)
            
            return {'success': True, 'message': 'Keywords updated successfully'}
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.post("/api/add-keyword")
async def add_keyword(data: dict = Body(...)):
    """添加新关键词并持久化到CSV"""
    global keywords_mapping
    keyword = data.get("keyword", "").strip()
    translation = data.get("translation", "").strip()
    if not keyword or not translation:
        return {"success": False, "error": "Keyword and translation required"}
    # 更新内存
    keywords_mapping[keyword.lower()] = translation
    # 追加到CSV（去重）
    import pandas as pd
    import os
    csv_path = 'data/keywords.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 检查是否已存在
        if not ((df['keyword'].str.lower() == keyword.lower()).any()):
            new_row = pd.DataFrame([{'keyword': keyword, 'translation': translation}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)
    else:
        df = pd.DataFrame([[keyword, translation]], columns=['keyword', 'translation'])
        df.to_csv(csv_path, index=False)
    return {"success": True, "message": "Keyword added"}

@app.get("/api/download-keywords")
def download_keywords():
    """下载当前关键词CSV文件"""
    import os
    csv_path = 'data/keywords.csv'
    if not os.path.exists(csv_path):
        # 返回空文件
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('keyword,translation\n')
    return FileResponse(csv_path, filename='keywords.csv', media_type='text/csv')

@app.post("/api/start-listening")
async def start_speech_recognition():
    """Start speech recognition"""
    global is_listening
    
    if not is_listening:
        start_listening()
        return {'success': True, 'message': 'Speech recognition started'}
    
    return {'success': False, 'message': 'Already listening'}

@app.post("/api/stop-listening")
async def stop_speech_recognition():
    """Stop speech recognition"""
    global is_listening
    stop_listening()
    return {'success': True, 'message': 'Speech recognition stopped'}

@app.post("/api/restart-whisper")
async def restart_whisper():
    try:
        # Stop listening if running
        global is_listening
        if is_listening:
            stop_listening()
        # Start listening
        start_listening()
        return {"success": True, "message": "Whisper restarted successfully."}
    except Exception as e:
        return {"success": False, "message": f"Failed to restart Whisper: {e}"}

@app.post("/api/reset-line")
async def reset_line():
    """Reset current line index"""
    global current_line_index
    current_line_index = 0
    return {'success': True, 'message': 'Line index reset'}

@app.post("/api/set-audio-device")
async def set_audio_device(device_data: AudioDeviceData):
    global selected_device_id
    try:
        # 支持 device_id 为 None 或 -1，表示系统默认
        if device_data.device_id is None or device_data.device_id == -1:
            selected_device_id = None
            save_config()
            return {'success': True, 'message': 'Audio device set to system default'}
        devices = sd.query_devices()
        if device_data.device_id < 0 or device_data.device_id >= len(devices):
            return {'success': False, 'error': 'Invalid device ID'}
        device = devices[device_data.device_id]
        if device['max_input_channels'] <= 0:
            return {'success': False, 'error': 'Device is not an input device'}
        selected_device_id = device_data.device_id
        save_config()
        return {'success': True, 'message': f'Audio device set to: {device["name"]}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.get("/api/audio-devices")
async def get_audio_devices():
    """Get available audio input devices"""
    try:
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Only input devices
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return {'success': True, 'devices': input_devices}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.get("/api/status")
async def get_status():
    return {
        'is_listening': is_listening,
        'current_line_index': current_line_index,
        'total_lines': len(script_translated),
        'model_loaded': model is not None,
        'current_model': selected_model_name,
        'recognition_language': RECOGNITION_LANGUAGE_UI,
        'match_threshold': MATCH_THRESHOLD,
        'audio_buffer_seconds': AUDIO_BUFFER_SECONDS,
        'silence_threshold': SILENCE_THRESHOLD,
        'silence_duration': SILENCE_DURATION,
        'model_cache_dir': model_cache_dir,
        'selected_device_id': selected_device_id,
        'selected_device_name': selected_device_name if 'selected_device_name' in globals() else None
    }

@app.get("/api/whisper-models")
async def get_whisper_models():
    """Get available Whisper models and their status"""
    import whisper
    from pathlib import Path
    
    available_models = whisper.available_models()
    
    # Model size estimates (in GB) - these are approximate sizes
    model_sizes = {
        'tiny': 0.039, 'tiny.en': 0.039,
        'base': 0.145, 'base.en': 0.145,
        'small': 0.466, 'small.en': 0.466,
        'medium': 1.42, 'medium.en': 1.42,
        'large': 2.87, 'large-v1': 2.87, 'large-v2': 2.87, 'large-v3': 2.87
    }

    # Use custom cache directory if set, otherwise use default
    if model_cache_dir:
        cache_dir = Path(model_cache_dir)
    else:
        cache_dir = Path.home() / ".cache" / "whisper"

    downloaded_models = []
    if cache_dir.exists():
        for file in cache_dir.glob("*.pt"):
            model_name = file.stem
            size_mb = file.stat().st_size / (1024 * 1024)
            downloaded_models.append({
                'name': model_name,
                'size_mb': round(size_mb, 1),
                'downloaded': True
            })
    # 新增：如果目录不存在，所有模型均未下载
    else:
        downloaded_models = []

    models = []
    for model_name in available_models:
        downloaded = any(m['name'] == model_name for m in downloaded_models)
        size_info = next((m for m in downloaded_models if m['name'] == model_name), None)
        estimated_size_gb = model_sizes.get(model_name, 0.0)
        models.append({
            'name': model_name,
            'downloaded': downloaded,
            'size_mb': size_info['size_mb'] if size_info else None,
            'estimated_size_gb': estimated_size_gb,
            'current': model_name == selected_model_name
        })

    return {
        'models': models,
        'current_model': selected_model_name,
        'cache_dir': str(cache_dir)
    }

@app.post("/api/set-config")
async def set_config(config: dict = Body(...)):
    try:
        global MATCH_THRESHOLD, AUDIO_BUFFER_SECONDS, SILENCE_THRESHOLD, SILENCE_DURATION, model_cache_dir, model, RECOGNITION_LANGUAGE_UI, selected_model_name
        updated = []
        errors = []
        # model_cache_dir
        if 'model_cache_dir' in config:
            value = config['model_cache_dir']
            import os
            from pathlib import Path
            cache_path = Path(value)
            if not cache_path.exists():
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f'Failed to create cache dir: {e}')
            if model_cache_dir != value:
                model = None
                model_cache_dir = value
                updated.append('model_cache_dir')
                print(f"Model cache directory changed to: {model_cache_dir}")
                save_config()
        # match_threshold
        if 'match_threshold' in config:
            value = config['match_threshold']
            if 0.0 < value <= 1.0:
                MATCH_THRESHOLD = value
                updated.append('match_threshold')
            else:
                errors.append('match_threshold must be between 0 and 1')
        # audio_buffer_seconds
        if 'audio_buffer_seconds' in config:
            value = config['audio_buffer_seconds']
            if 0.5 <= value <= 10.0:
                AUDIO_BUFFER_SECONDS = value
                updated.append('audio_buffer_seconds')
            else:
                errors.append('audio_buffer_seconds must be between 0.5 and 10.0')
        # silence_threshold
        if 'silence_threshold' in config:
            value = config['silence_threshold']
            if 0.001 <= value <= 0.1:
                SILENCE_THRESHOLD = value
                updated.append('silence_threshold')
            else:
                errors.append('silence_threshold must be between 0.001 and 0.1')
        # silence_duration
        if 'silence_duration' in config:
            value = config['silence_duration']
            if 0.05 <= value <= 2.0:
                SILENCE_DURATION = value
                updated.append('silence_duration')
            else:
                errors.append('silence_duration must be between 0.05 and 2.0')
        # selected_model_name（迁移自 /api/set-model）
        if 'selected_model_name' in config:
            value = config['selected_model_name']
            import whisper
            if value not in whisper.available_models():
                errors.append('Invalid model name')
            else:
                if selected_model_name != value:
                    model = None
                    selected_model_name = value
                    updated.append('selected_model_name')
                    print(f"Model changed to: {selected_model_name}")
                save_config()
        # 其它参数...
        if updated:
            save_config()
        if errors:
            return {"success": False, "updated": updated, "errors": errors}
        return {"success": True, "updated": updated}
    except Exception as e:
        import traceback
        print("[ERROR] Exception in /api/set-config:", e)
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/api/download-model")
async def download_model(data: ModelData):
    """Download a specific Whisper model"""
    global download_progress, active_download_threads
    
    try:
        import whisper
        import subprocess
        import sys
        import threading
        import time
        
        # Validate model name
        if data.model_name not in whisper.available_models():
            return {'success': False, 'error': 'Invalid model name'}
        
        # Initialize progress
        download_progress[data.model_name] = {
            'status': 'starting',
            'progress': 0,
            'message': f'Starting download of {data.model_name}...'
        }
        
        def download_with_progress():
            try:
                # Update status to downloading
                download_progress[data.model_name]['status'] = 'downloading'
                download_progress[data.model_name]['progress'] = 5
                download_progress[data.model_name]['message'] = f'Initializing download for {data.model_name}...'
                time.sleep(0.3)
                for progress in [15, 25, 35, 45, 55, 65, 75]:
                    download_progress[data.model_name]['progress'] = progress
                    download_progress[data.model_name]['message'] = f'Downloading {data.model_name}... ({progress}%)'
                    time.sleep(0.2)
                # 直接import并调用
                import download_whisper_models
                ok = download_whisper_models.download_model_by_name(data.model_name, model_cache_dir, quiet=True)
                if ok:
                    download_progress[data.model_name]['progress'] = 85
                    download_progress[data.model_name]['message'] = f'Finalizing {data.model_name}...'
                    time.sleep(0.3)
                    download_progress[data.model_name]['status'] = 'completed'
                    download_progress[data.model_name]['progress'] = 100
                    download_progress[data.model_name]['message'] = f'Model {data.model_name} downloaded successfully'
                else:
                    download_progress[data.model_name]['status'] = 'error'
                    download_progress[data.model_name]['message'] = f'Download failed: see server logs for details'
            except Exception as e:
                download_progress[data.model_name]['status'] = 'error'
                download_progress[data.model_name]['message'] = f'Download error: {str(e)}'
            finally:
                # 下载结束后移除线程记录
                if data.model_name in active_download_threads:
                    del active_download_threads[data.model_name]
        
        # Start download in background thread
        thread = threading.Thread(target=download_with_progress)
        thread.daemon = True
        thread.start()
        active_download_threads[data.model_name] = thread
        
        return {'success': True, 'message': f'Download started for {data.model_name}'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.post("/api/stop-download/{model_name}")
async def stop_download(model_name: str):
    """Stop downloading a specific model"""
    global active_download_threads, download_progress
    thread = active_download_threads.get(model_name)
    if thread and thread.is_alive():
        # Python没有安全的线程kill，只能设置状态让下载函数主动退出
        # 这里我们用download_progress的特殊标记让下载函数检测到后退出
        download_progress[model_name]['status'] = 'stopping'
        download_progress[model_name]['message'] = 'Stopping...'
        return {'success': True, 'message': f'Stopping download for {model_name}'}
    else:
        return {'success': False, 'message': f'No active download for {model_name}'}

@app.get("/api/download-progress/{model_name}")
async def get_download_progress(model_name: str):
    """Get download progress for a specific model"""
    global download_progress
    
    if model_name in download_progress:
        return download_progress[model_name]
    else:
        return {'status': 'not_found', 'progress': 0, 'message': 'No download in progress'}

@app.get("/api/match-threshold")
async def get_match_threshold():
    return {"match_threshold": MATCH_THRESHOLD}

@app.post("/api/match-threshold")
async def set_match_threshold(threshold: float = Body(..., embed=True)):
    global MATCH_THRESHOLD
    if not (0.0 < threshold <= 1.0):
        return {"success": False, "error": "Threshold must be between 0 and 1."}
    MATCH_THRESHOLD = threshold
    save_config()
    return {"success": True, "match_threshold": MATCH_THRESHOLD}

@app.get("/api/recognition-language")
async def get_recognition_language():
    lang = RECOGNITION_LANGUAGE_UI
    allowed_langs = ["en", "ja", "ko", "es", "fr", "de", "ru", "it", "pt", "ar", "hi", "auto"]
    if lang not in allowed_langs:
        lang = "auto"
    return {"recognition_language": lang}

@app.post("/api/set-recognition-language")
async def set_recognition_language(data: dict = Body(...)):
    """设置识别语言（支持常用非中文语言）"""
    global RECOGNITION_LANGUAGE, RECOGNITION_LANGUAGE_UI
    lang = data.get("language")
    allowed_langs = ["en", "ja", "ko", "es", "fr", "de", "ru", "it", "pt", "ar", "hi", "auto", None]
    if lang in allowed_langs:
        RECOGNITION_LANGUAGE_UI = lang
        # 持久化到配置
        save_config()
        return {"success": True, "message": f"Recognition language set to {lang}"}
    else:
        return {"success": False, "error": "Unsupported language (Chinese is not supported)"}

@app.get("/api/audio-buffer-seconds")
async def get_audio_buffer_seconds():
    return {"audio_buffer_seconds": AUDIO_BUFFER_SECONDS}

@app.post("/api/audio-buffer-seconds")
async def set_audio_buffer_seconds(seconds: float = Body(..., embed=True)):
    global AUDIO_BUFFER_SECONDS
    if not (0.5 <= seconds <= 10.0):
        return {"success": False, "error": "Buffer seconds must be between 0.5 and 10."}
    AUDIO_BUFFER_SECONDS = seconds
    save_config()
    return {"success": True, "audio_buffer_seconds": AUDIO_BUFFER_SECONDS}

@app.get("/api/silence-config")
async def get_silence_config():
    global SILENCE_THRESHOLD, SILENCE_DURATION
    return {
        "silence_threshold": SILENCE_THRESHOLD,
        "silence_duration": SILENCE_DURATION
    }

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    print("[DEBUG] WebSocket connection attempt from frontend")
    await manager.connect(websocket)
    # 首次连接时推送当前内容
    if manager.last_push_data:
        await websocket.send_json(manager.last_push_data)
    try:
        while True:
            await asyncio.sleep(60)  # 保持连接心跳
    except WebSocketDisconnect:
        print("[DEBUG] WebSocket disconnected")
        manager.disconnect(websocket)

def load_cache_config():
    global model_cache_dir, selected_model_name, MATCH_THRESHOLD, RECOGNITION_LANGUAGE, AUDIO_BUFFER_SECONDS, RECOGNITION_LANGUAGE_UI, SILENCE_THRESHOLD, SILENCE_DURATION, selected_device_id
    global selected_device_name
    try:
        import json
        if os.path.exists('cache_config.json'):
            with open('cache_config.json', 'r') as f:
                config = json.load(f)
                if 'model_cache_dir' in config:
                    model_cache_dir = config['model_cache_dir']
                    print(f"📁 加载缓存目录配置: {model_cache_dir}")
                if 'selected_model_name' in config:
                    selected_model_name = config['selected_model_name']
                    print(f"🤖 加载模型配置: {selected_model_name}")
                if 'match_threshold' in config:
                    MATCH_THRESHOLD = config['match_threshold']
                    print(f"🔎 加载匹配阈值: {MATCH_THRESHOLD}")
                if 'recognition_language' in config:
                    RECOGNITION_LANGUAGE_UI = config['recognition_language']
                    print(f"🌐 加载识别语言: {RECOGNITION_LANGUAGE_UI}")
                if 'audio_buffer_seconds' in config:
                    AUDIO_BUFFER_SECONDS = config['audio_buffer_seconds']
                    print(f"🎤 加载音频缓冲区时长: {AUDIO_BUFFER_SECONDS}")
                if 'silence_threshold' in config:
                    SILENCE_THRESHOLD = config['silence_threshold']
                    print(f"🔇 加载静音阈值: {SILENCE_THRESHOLD}")
                if 'silence_duration' in config:
                    SILENCE_DURATION = config['silence_duration']
                    print(f"🔇 加载静音时长: {SILENCE_DURATION}")
                # 新增设备ID和设备名
                if 'selected_device_id' in config:
                    val = config['selected_device_id']
                    if val is not None:
                        selected_device_id = int(val)
                        print(f"🎤 加载音频设备ID: {selected_device_id}")
                    else:
                        selected_device_id = None
                if 'selected_device_name' in config:
                    selected_device_name = config['selected_device_name']
                    print(f"🎤 加载音频设备名: {selected_device_name}")
    except Exception as e:
        print(f"⚠️ 加载缓存配置失败: {e}")

def save_config():
    try:
        device_name = None
        if selected_device_id is not None:
            try:
                devices = sd.query_devices()
                if 0 <= selected_device_id < len(devices):
                    device_name = devices[selected_device_id]['name']
            except Exception:
                pass
        config = {
            'model_cache_dir': model_cache_dir,
            'selected_model_name': selected_model_name,
            'match_threshold': MATCH_THRESHOLD,
            'recognition_language': RECOGNITION_LANGUAGE_UI,
            'audio_buffer_seconds': AUDIO_BUFFER_SECONDS,
            'silence_threshold': SILENCE_THRESHOLD,
            'silence_duration': SILENCE_DURATION,
            'selected_device_id': int(selected_device_id) if selected_device_id is not None else None,
            'selected_device_name': device_name
        }
        with open('cache_config.json', 'w') as f:
            import json
            json.dump(config, f, indent=4)
        print(f"Configuration saved to cache_config.json")
    except Exception as e:
        import traceback
        print(f"⚠️ 保存配置失败: {e}")
        traceback.print_exc()

# Startup event
@app.on_event("startup")
async def startup_event():
    global MAIN_LOOP
    import asyncio
    MAIN_LOOP = asyncio.get_running_loop()
    print("[DEBUG] FastAPI startup event triggered")
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load cache configuration
    load_cache_config()
    
    # Load scripts and keywords
    load_scripts()
    load_keywords()
    
    print("Voice-synced Text Display System")
    print("Access the display at: http://localhost:8080")
    print("Access the admin panel at: http://localhost:8080/admin")
    print("API documentation at: http://localhost:8080/docs")

@app.get("/api/torch-device")
def get_torch_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"device": device}

@app.get("/api/audio-device-config")
async def get_audio_device_config():
    return {
        "selected_device_id": selected_device_id,
        "selected_device_name": selected_device_name if 'selected_device_name' in globals() else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 