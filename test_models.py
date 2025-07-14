#!/usr/bin/env python3
"""
测试模型管理功能
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_get_models():
    """测试获取模型列表"""
    print("🔍 测试获取模型列表...")
    try:
        response = requests.get(f"{BASE_URL}/api/whisper-models")
        if response.status_code == 200:
            data = response.json()
            print("✅ 成功获取模型列表")
            print(f"   当前模型: {data['current_model']}")
            print(f"   缓存目录: {data['cache_dir']}")
            print(f"   模型数量: {len(data['models'])}")
            
            # 显示已下载的模型
            downloaded = [m for m in data['models'] if m['downloaded']]
            print(f"   已下载模型: {len(downloaded)}")
            for model in downloaded:
                print(f"     - {model['name']} ({model['size_mb']}MB)")
            
            return True
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取模型列表出错: {e}")
        return False

def test_set_model():
    """测试设置模型"""
    print("\n🔧 测试设置模型...")
    try:
        # 先获取当前模型
        response = requests.get(f"{BASE_URL}/api/whisper-models")
        if response.status_code == 200:
            data = response.json()
            current_model = data['current_model']
            
            # 尝试切换到small模型
            test_model = "small"
            if test_model == current_model:
                test_model = "base"  # 如果当前是small，切换到base
            
            print(f"   从 {current_model} 切换到 {test_model}")
            
            response = requests.post(
                f"{BASE_URL}/api/set-model",
                headers={"Content-Type": "application/json"},
                json={"model_name": test_model}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    print(f"✅ 成功设置模型为 {test_model}")
                    return True
                else:
                    print(f"❌ 设置模型失败: {data['error']}")
                    return False
            else:
                print(f"❌ 设置模型请求失败: {response.status_code}")
                return False
        else:
            print("❌ 无法获取当前模型信息")
            return False
    except Exception as e:
        print(f"❌ 设置模型出错: {e}")
        return False

def test_set_cache_dir():
    """测试设置缓存目录"""
    print("\n📁 测试设置缓存目录...")
    try:
        test_cache_dir = "/tmp/whisper_test"
        
        response = requests.post(
            f"{BASE_URL}/api/set-model-cache",
            headers={"Content-Type": "application/json"},
            json={"cache_dir": test_cache_dir}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ 成功设置缓存目录为 {test_cache_dir}")
                return True
            else:
                print(f"❌ 设置缓存目录失败: {data['error']}")
                return False
        else:
            print(f"❌ 设置缓存目录请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 设置缓存目录出错: {e}")
        return False

def test_download_model():
    """测试下载模型"""
    print("\n⬇️ 测试下载模型...")
    try:
        # 尝试下载tiny.en模型（通常较小）
        test_model = "tiny.en"
        
        print(f"   尝试下载 {test_model}...")
        
        response = requests.post(
            f"{BASE_URL}/api/download-model",
            headers={"Content-Type": "application/json"},
            json={"model_name": test_model}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ 成功下载模型 {test_model}")
                return True
            else:
                print(f"❌ 下载模型失败: {data['error']}")
                return False
        else:
            print(f"❌ 下载模型请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 下载模型出错: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试模型管理功能")
    print("=" * 50)
    
    tests = [
        test_get_models,
        test_set_model,
        test_set_cache_dir,
        test_download_model
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试执行出错: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    test_names = ["获取模型列表", "设置模型", "设置缓存目录", "下载模型"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {i+1}. {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！模型管理功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")

if __name__ == "__main__":
    main() 