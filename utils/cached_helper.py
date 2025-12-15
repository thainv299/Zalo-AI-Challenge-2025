# Cache VLM vào disk
import csv
import hashlib
import json
import os

def save_temp_results(results, temp_file_path):
    """Lưu kết quả tạm thời"""
    sorted_results = sorted(results, key=lambda x: x['index'])
    csv_data = [{'id': r['id'], 'answer': r['answer']} for r in sorted_results]
    
    with open(temp_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"💾 Backup: {len(results)} kết quả -> {temp_file_path}")

def get_vlm_cache(video_path):
    """Load VLM description và video_info từ cache"""
    cache_dir = "cached_vlm"
    os.makedirs(cache_dir, exist_ok=True)
    # Sử dụng tên file trực tiếp thay vì hash
    video_name = os.path.basename(video_path).replace('.mp4', '')
    cache_file = os.path.join(cache_dir, f"{video_name}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('vlm_description'), data.get('video_info', '')
    return None, None

def save_vlm_cache(video_path, vlm_description, video_info):
    """Lưu VLM description và video_info vào cache"""
    cache_dir = "cached_vlm"
    os.makedirs(cache_dir, exist_ok=True)
    # Sử dụng tên file trực tiếp thay vì hash
    video_name = os.path.basename(video_path).replace('.mp4', '')
    cache_file = os.path.join(cache_dir, f"{video_name}.json")
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'vlm_description': vlm_description,
            'video_info': video_info
        }, f, ensure_ascii=False)

def save_json(data, filename):
    cached_dir = "cached_helper"    
    os.makedirs(cached_dir, exist_ok=True)
    with open(os.path.join(cached_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
