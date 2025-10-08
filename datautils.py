import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    # try:
        # 首先尝试本地路径
    data_dir = "/mnt/share/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw"
    # /mnt/share/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw
    def load_split(split_name):
        # 拼接完整文件路径（如：/mnt/.../wikitext-2-raw-train.txt）wiki.test.raw
        file_path = os.path.join(data_dir, f"wiki.{split_name}.raw")
        # 检查文件是否存在（避免路径错误）
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到本地文件：{file_path}\n请检查路径是否正确！")
        # 读取并过滤无效行
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 过滤空行、仅空格的行，以及以 "=" 开头的标题行（Wikitext 数据集特点）
        valid_texts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("=")]
        return valid_texts
    # 3. 加载训练集和测试集（不再用 load_dataset）
    train_texts = load_split("train")  # 加载训练集文本
    test_texts = load_split("test")  # 加载测试集文本
    # except ValueError:
    #     # 如果本地路径失败，则使用在线加载
    #     traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    #     testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(train_texts), return_tensors='pt')
    testenc = tokenizer("\n\n".join(test_texts), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    # 从本地路径加载C4数据集
    data_dir = "/mnt/share/Datasets/LLMC/c4"
    
    # 检查本地路径是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到本地C4数据集：{data_dir}\n请检查路径是否正确！")
    
    # 直接从本地目录加载数据集（不需要指定格式）
    from datasets import Dataset
    dataset = Dataset.from_file(os.path.join(data_dir, 'data-00000-of-00002.arrow'))
    
    # 使用同一个数据集作为训练和验证
    traindata = dataset
    valdata = dataset

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # 对验证集采样，避免数据量过大
    val_samples = min(1100, len(valdata))
    valenc = tokenizer(' '.join(valdata[:val_samples]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
