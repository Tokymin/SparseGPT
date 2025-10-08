#!/usr/bin/env python3
"""
快速测试脚本 - 验证 SparseGPT 运行状态
"""

import torch
import time
from opt_toky import get_opt
from datautils import get_loaders


def quick_test():
    """快速测试当前模型状态"""
    print("🔍 快速状态检查...")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔢 CUDA版本: {torch.version.cuda}")
    
    # 测试模型加载
    print("\n📥 测试模型加载...")
    try:
        model = get_opt("facebook/opt-125m")
        print(f"✅ 模型加载成功: {model.config.model_type}")
        print(f"📊 参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🏗️  层数: {len(model.model.decoder.layers)}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试数据加载
    print("\n📊 测试数据加载...")
    try:
        dataloader, testloader = get_loaders(
            "wikitext2", nsamples=32, seed=0,
            model="facebook/opt-125m", seqlen=model.seqlen
        )
        print(f"✅ 数据加载成功")
        print(f"📈 校准样本数: {len(dataloader)}")
        print(f"🧪 测试样本数: {testloader.input_ids.shape}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 测试前向传播
    print("\n🚀 测试前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            # 测试一个小批次
            sample_input = torch.randint(0, 1000, (1, 10)).to(device)
            start_time = time.time()
            output = model(sample_input)
            inference_time = time.time() - start_time
            print(f"✅ 前向传播成功")
            print(f"⏱️  推理时间: {inference_time:.3f}秒")
            print(f"📊 输出形状: {output.logits.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return
    
    # 检查稀疏化状态
    print("\n🗜️  检查当前稀疏化状态...")
    total_params = 0
    sparse_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            sparse_params += torch.sum(param == 0).item()
    
    if total_params > 0:
        current_sparsity = sparse_params / total_params
        print(f"📊 当前稀疏度: {current_sparsity:.1%}")
        print(f"🔢 稀疏参数: {sparse_params:,}")
        print(f"📈 总参数: {total_params:,}")
    else:
        print("⚠️  未找到权重参数")
    
    # 内存使用情况
    print("\n💾 内存使用情况...")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"🎮 GPU已分配: {allocated:.2f} GB")
        print(f"💾 GPU已缓存: {cached:.2f} GB")
    
    print("\n✅ 快速测试完成！")
    print("💡 提示: 如果所有测试都通过，您的 SparseGPT 环境配置正确")


if __name__ == "__main__":
    quick_test()
