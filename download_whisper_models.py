#!/usr/bin/env python3
"""
Whisper Models Download Script
用于下载所有Whisper模型以实现离线使用
"""

import whisper
import os
import sys
from pathlib import Path

def get_cache_dir(cache_dir=None):
    """获取Whisper缓存目录（自定义目录时不再拼接whisper子目录）"""
    if cache_dir:
        os.environ['XDG_CACHE_HOME'] = str(Path(cache_dir).parent)
        return Path(cache_dir)
    if 'XDG_CACHE_HOME' in os.environ:
        cache_dir = Path(os.environ['XDG_CACHE_HOME']) / "whisper"
    else:
        cache_dir = Path.home() / ".cache" / "whisper"
    return cache_dir

def check_existing_models(cache_dir=None):
    """检查已下载的模型"""
    cache_dir = get_cache_dir(cache_dir)
    existing_models = []
    if cache_dir.exists():
        for file in cache_dir.glob("*.pt"):
            model_name = file.stem
            size_mb = file.stat().st_size / (1024 * 1024)
            existing_models.append((model_name, size_mb))
    return existing_models

def download_model_by_name(model_name, cache_dir=None, quiet=False):
    from pathlib import Path
    cache_dir_path = Path(cache_dir) if cache_dir else None
    if cache_dir_path:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"📁 缓存目录: {cache_dir_path if cache_dir_path else '[默认] ~/.cache/whisper'}")
        print(f"🎯 准备下载模型: {model_name}")
    model_file = (cache_dir_path / f"{model_name}.pt") if cache_dir_path else None
    if model_file and model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        if size_mb < 1.0:
            # 文件太小，视为无效，删除
            if not quiet:
                print(f"   ⚠️ 检测到异常模型文件（{size_mb:.2f}MB），自动删除重新下载")
            model_file.unlink()
        else:
            if not quiet:
                print(f"   ✅ 模型已存在 ({size_mb:.1f}MB)")
            return True
    try:
        if not quiet:
            print(f"   ⏳ 正在下载...")
        import whisper
        if cache_dir_path:
            whisper.load_model(model_name, download_root=str(cache_dir_path))
        else:
            whisper.load_model(model_name)
        if model_file and model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            if size_mb < 1.0:
                if not quiet:
                    print(f"   ❌ 下载失败，文件过小（{size_mb:.2f}MB），自动删除")
                model_file.unlink()
                return False
            if not quiet:
                print(f"   ✅ 下载完成 ({size_mb:.1f}MB)")
            return True
        elif not cache_dir_path:
            # 检查默认目录
            from pathlib import Path
            default_file = Path.home() / ".cache" / "whisper" / f"{model_name}.pt"
            if default_file.exists():
                size_mb = default_file.stat().st_size / (1024 * 1024)
                if size_mb < 1.0:
                    if not quiet:
                        print(f"   ❌ 下载失败，文件过小（{size_mb:.2f}MB），自动删除")
                    default_file.unlink()
                    return False
                if not quiet:
                    print(f"   ✅ 下载完成 ({size_mb:.1f}MB)")
                return True
        return False
    except Exception as e:
        if model_file and model_file.exists():
            model_file.unlink()
        if not quiet:
            print(f"   ❌ 下载出错: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Whisper模型下载工具')
    parser.add_argument('--model', type=str, help='指定要下载的模型名称')
    parser.add_argument('--cache-dir', type=str, help='指定模型缓存目录')
    parser.add_argument('--list', action='store_true', help='列出所有可用模型')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    args = parser.parse_args()
    if args.cache_dir:
        os.environ['XDG_CACHE_HOME'] = args.cache_dir
        if not args.quiet:
            print(f"📁 使用自定义缓存目录: {args.cache_dir}")
    if args.model:
        if not args.quiet:
            print(f"🚀 开始下载模型: {args.model}")
        ok = download_model_by_name(args.model, args.cache_dir, args.quiet)
        if not args.quiet:
            print("✅ 下载完成！" if ok else "❌ 下载失败！")
        sys.exit(0 if ok else 1)
    if args.list:
        available_models = whisper.available_models()
        existing_models = check_existing_models(args.cache_dir)
        existing_names = [name for name, _ in existing_models]
        for model in available_models:
            status = "✅" if model in existing_names else "❌"
            print(f"{status} {model}")
        sys.exit(0)
    # 交互模式略

if __name__ == "__main__":
    main() 