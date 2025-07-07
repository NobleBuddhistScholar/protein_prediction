#!/usr/bin/env python3
"""
测试报告管理功能的脚本
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_get_all_summaries():
    """测试获取所有报告"""
    print("=== 测试获取所有报告 ===")
    try:
        response = requests.get(f"{BASE_URL}/getAllSummaries")
        if response.status_code == 200:
            summaries = response.json()
            print(f"✓ 成功获取 {len(summaries)} 个报告")
            for summary in summaries:
                print(f"  - {summary}")
            return summaries
        else:
            print(f"✗ 获取失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return []

def test_get_report_details(report_filename):
    """测试获取报告详情"""
    print(f"\n=== 测试获取报告详情: {report_filename} ===")
    try:
        response = requests.get(f"{BASE_URL}/reports/{report_filename}/details")
        if response.status_code == 200:
            details = response.json()
            print("✓ 成功获取报告详情:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            return details
        else:
            print(f"✗ 获取失败: {response.status_code}")
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            print(f"  错误信息: {error_data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return None

def test_delete_report(report_filename):
    """测试删除报告"""
    print(f"\n=== 测试删除报告: {report_filename} ===")
    
    # 先确认报告存在
    details = test_get_report_details(report_filename)
    if not details:
        print("✗ 报告不存在，跳过删除测试")
        return False
    
    # 确认删除
    confirm = input(f"确定要删除报告 '{report_filename}' 吗？ (y/n): ")
    if confirm.lower() != 'y':
        print("取消删除")
        return False
    
    try:
        response = requests.delete(f"{BASE_URL}/reports/{report_filename}")
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 删除成功: {result.get('message')}")
            return True
        else:
            print(f"✗ 删除失败: {response.status_code}")
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            print(f"  错误信息: {error_data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False

def test_download_report(report_filename):
    """测试下载报告"""
    print(f"\n=== 测试下载报告: {report_filename} ===")
    try:
        response = requests.get(f"{BASE_URL}/downloadSummary/{report_filename}")
        if response.status_code == 200:
            print("✓ 下载成功")
            print(f"  内容长度: {len(response.text)} 字符")
            print(f"  前100字符: {response.text[:100]}...")
            return True
        else:
            print(f"✗ 下载失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False

def main():
    print("基因组分析报告管理功能测试")
    print("=" * 50)
    
    # 获取所有报告
    summaries = test_get_all_summaries()
    
    if not summaries:
        print("\n没有找到报告文件，测试结束")
        return
    
    # 选择一个报告进行详细测试
    if len(summaries) > 0:
        test_report = summaries[0]
        
        # 测试详情
        test_get_report_details(test_report)
        
        # 测试下载
        test_download_report(test_report)
        
        # 测试删除（需要用户确认）
        test_delete_report(test_report)
        
        # 再次获取列表确认删除结果
        print("\n=== 删除后重新获取报告列表 ===")
        test_get_all_summaries()

if __name__ == "__main__":
    main()
