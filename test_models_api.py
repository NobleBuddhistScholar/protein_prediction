#!/usr/bin/env python3
"""
测试模型选择API功能
"""

import requests
import json
import sys

def test_models_api():
    """测试模型列表API"""
    try:
        # 测试获取模型列表
        print("测试获取模型列表...")
        response = requests.get('http://localhost:5000/models')
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 模型列表获取成功:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ 模型列表获取失败: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败: 请确保Flask服务器正在运行 (python api.py)")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_models_api()
    sys.exit(0 if success else 1)
