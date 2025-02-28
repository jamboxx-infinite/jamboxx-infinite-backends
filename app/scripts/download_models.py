import os
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm
import yaml

# 模型信息配置
MODEL_INFO = {
    'hubert_soft.pt': {
        'url': 'https://example.com/models/hubert_soft.pt',
        'md5': 'your-md5-hash-here',
        'path': 'models/hubert/hubert_soft.pt'
    },
    'rmvpe.pt': {
        'url': 'https://example.com/models/rmvpe.pt',
        'md5': 'your-md5-hash-here',
        'path': 'models/rmvpe/rmvpe.pt'
    },
    'ddsp_model.pt': {
        'url': 'https://example.com/models/ddsp_model.pt',
        'md5': 'your-md5-hash-here',
        'path': 'models/ddsp/model.pt'
    },
    'nsf_hifigan.pt': {
        'url': 'https://example.com/models/nsf_hifigan.pt',
        'md5': 'your-md5-hash-here',
        'path': 'models/enhancer/nsf_hifigan.pt'
    }
}

def check_md5(file_path, expected_md5):
    """检查文件MD5是否匹配"""
    with open(file_path, 'rb') as f:
        file_md5 = hashlib.md5(f.read()).hexdigest()
    return file_md5 == expected_md5

def download_file(url, dest_path):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    """主函数"""
    print("开始下载模型文件...")
    
    for model_name, info in MODEL_INFO.items():
        dest_path = info['path']
        
        # 检查文件是否已存在且MD5匹配
        if os.path.exists(dest_path) and check_md5(dest_path, info['md5']):
            print(f"{model_name} 已存在且验证通过，跳过下载")
            continue
            
        print(f"\n下载 {model_name}...")
        try:
            download_file(info['url'], dest_path)
            if check_md5(dest_path, info['md5']):
                print(f"{model_name} 下载完成并验证通过")
            else:
                print(f"警告：{model_name} MD5校验失败！")
        except Exception as e:
            print(f"下载 {model_name} 时出错：{str(e)}")

if __name__ == "__main__":
    main()