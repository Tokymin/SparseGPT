SparseGPT
本仓库包含用于复现论文 SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot 关键结果的代码。

具体而言，它提供了以下脚本和实现：

在 raw-WikiText2、PTB 和 C4 子集上评估基线模型和剪枝模型。（datautils.py、opt.py、bloom.py）
在 OPT 和 BLOOM 模型上执行非结构化、n:m 以及稀疏+量化的 SparseGPT 压缩。（sparsegpt.py、opt.py、bloom.py）
请注意，此 SparseGPT 实现基于我们的开源 GPTQ 代码。

依赖项

-torch：在 v1.10.1+cu111 版本上测试通过

-transformers：在 v4.21.2 版本上测试通过

-datasets：在 v1.17.0 版本上测试通过

用法

以下是一些用于在 OPT 模型上运行基线模型和稀疏化，然后在 raw-WikiText2、PTB 和 C4 上进行困惑度评估的示例命令。另请参见命令行参数文档。


# 运行密集型基线模型
python opt.py facebook/opt-125m c4

# 运行基于幅度的基线模型
python opt.py facebook/opt-125m c4 --sparsity .5 --gmp

# 使用 SparseGPT 剪枝至 50% 均匀稀疏度
python opt.py facebook/opt-125m c4 --sparsity .5

# 使用 SparseGPT 剪枝至完全 2:4 稀疏度
python opt.py facebook/opt-125m c4 --prunen 2 --prunem 4

# 使用 SparseGPT 剪枝至 50% 稀疏度并进行 4 位量化
python opt.py facebook/opt-125m c4 --sparsity .5 --wbits 4
要在其他 OPT 模型上运行，请将 "facebook/opt-125m" 替换为相应模型的 HuggingFace 名称。对于 175B 模型，必须先向 Meta 申请访问权限并将检查点转换为 HuggingFace 格式，然后只需将其位置作为名称传递给此脚本即可。

BLOOM 脚本 bloom.py 具有非常相似的接口，但目前某些功能仅适用于 OPT，例如：


# 使用 SparseGPT 稀疏化 BLOOM-176B 模型
python bloom.py bigscience/bloom c4 --sparsity .5
我们还提供了具有相同接口的 LLaMA 剪枝脚本：


# 使用 SparseGPT 稀疏化 LLaMa 模型
python llama.py LLAMA_HF_WEIGHTS_LOCATION c4 --sparsity 0.5
如果想要保存稀疏化后的模型，可以通过 --save 标志指定保存检查点的路径。

还可以选择使用 --log_wandb 将评估结果记录到 W&B。

演示
可以通过 colab 演示 demo.ipynb 尝试 SparseGPT。

引用
如果您发现本工作有用，请考虑引用：


@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}