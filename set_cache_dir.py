#!/usr/bin/env python3
"""
设置Whisper缓存目录脚本
"""

import requests
import json
import sys

def set_cache_dir(cache_dir):
    """设置Whisper缓存目录"""
    try:
        response = requests.post(
            'http://localhost:8000/api/set-model-cache',
            headers={'Content-Type': 'application/json'},
            json={'cache_dir': cache_dir}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ 成功设置缓存目录: {cache_dir}")
                return True
            else:
                print(f"❌ 设置失败: {data['error']}")
                return False
        else:
            print(f"❌ 请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 设置缓存目录出错: {e}")
        return False

def check_models():
    """检查模型状态"""
    try:
        response = requests.get('http://localhost:8000/api/whisper-models')
        
        if response.status_code == 200:
            data = response.json()
            print(f"📁 当前缓存目录: {data['cache_dir']}")
            print(f"🎯 当前模型: {data['current_model']}")
            
            downloaded = [m for m in data['models'] if m['downloaded']]
            print(f"📦 已下载模型: {len(downloaded)}")
            
            for model in downloaded:
                print(f"   ✅ {model['name']} ({model['size_mb']}MB)")
            
            return True
        else:
            print(f"❌ 获取模型信息失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 检查模型状态出错: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python set_cache_dir.py <缓存目录路径>")
        print("示例: python set_cache_dir.py /Volumes/1T/LargeApplications/AIModels/Whisper")
        return
    
    cache_dir = sys.argv[1]
    
    print("🔧 设置Whisper缓存目录")
    print("=" * 50)
    
    # 设置缓存目录
    if set_cache_dir(cache_dir):
        print("\n📋 检查模型状态:")
        check_models()
    else:
        print("❌ 设置缓存目录失败")

if __name__ == "__main__":
    main() 