#!/usr/bin/env python3
"""
测试模型管理API功能
"""

import requests
import json
import sys

BASE_URL = 'http://localhost:5000'

def test_get_models():
    """测试获取模型列表"""
    print("1. 测试获取模型列表...")
    try:
        response = requests.get(f'{BASE_URL}/models')
        if response.status_code == 200:
            data = response.json()
            print("✅ 获取模型列表成功:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return data.get('models', [])
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return []

def test_get_model_details(filename):
    """测试获取模型详情"""
    print(f"\n2. 测试获取模型详情: {filename}")
    try:
        response = requests.get(f'{BASE_URL}/models/{filename}/details')
        if response.status_code == 200:
            data = response.json()
            print("✅ 获取模型详情成功:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ 获取模型详情失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return False

def test_rename_model(old_name, new_name):
    """测试重命名模型"""
    print(f"\n3. 测试重命名模型: {old_name} -> {new_name}")
    try:
        response = requests.put(
            f'{BASE_URL}/models/{old_name}/rename',
            json={'new_name': new_name}
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ 重命名模型成功:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return data.get('new_filename')
        else:
            print(f"❌ 重命名模型失败: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return None

def test_delete_model(filename):
    """测试删除模型"""
    print(f"\n4. 测试删除模型: {filename}")
    try:
        response = requests.delete(f'{BASE_URL}/models/{filename}')
        if response.status_code == 200:
            data = response.json()
            print("✅ 删除模型成功:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ 删除模型失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return False

def main():
    print("开始测试模型管理API...")
    print("=" * 50)
    
    # 测试获取模型列表
    models = test_get_models()
    if not models:
        print("没有找到模型，跳过其他测试")
        return
    
    # 选择第一个模型进行测试
    test_model = models[0]['filename']
    
    # 测试获取模型详情
    test_get_model_details(test_model)
    
    # 注意：以下测试会修改实际文件，请谨慎使用
    print("\n" + "=" * 50)
    print("⚠️  警告：以下测试会修改实际文件")
    print("如果要继续测试重命名和删除功能，请手动运行相应的测试函数")
    
    # 如果你想测试重命名和删除，可以取消下面的注释
    # 但请确保你有备份或这些是测试文件
    
    # # 测试重命名（使用一个不存在的文件名避免影响实际文件）
    # new_filename = test_rename_model('test_model.pth', 'renamed_test_model')
    # if new_filename:
    #     # 测试删除重命名后的文件
    #     test_delete_model(new_filename)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        sys.exit(1)
