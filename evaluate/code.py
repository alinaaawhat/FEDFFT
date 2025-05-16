import os
import torch
import json
import numpy as np
import transformers
from transformers import GenerationConfig
from tqdm import tqdm
import gzip
import requests
from typing import Dict, List, Optional, Union, Tuple
import subprocess
import re



def clean_code(code):
    """
    清理生成的代码答案
    """
    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    # 1. 移除特殊字符 \u00a0
    code = code.replace('\u00a0', '')
    
    # 2. 移除停止序列后的所有内容
    for stop_seq in ['\nclass', '\ndef', '\n#', '\nif', '\nprint', '\nassert']:
        parts = code.split(stop_seq)
        if len(parts) > 1:
            code = parts[0]
    
    # 3. 添加适当的缩进以避免缩进错误
    lines = code.split('\n')
    padded_lines = []
    for line in lines:
        padded_lines.append(pad_spaces(line, 4))
    code = '\n'.join(padded_lines)
    
    return code

def download_humaneval_dataset(save_dir: str) -> str:
    """
    下载HumanEval数据集
    
    Args:
        save_dir: 保存数据集的目录
        
    Returns:
        文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'HumanEval.jsonl.gz')
    
    if not os.path.exists(filepath):
        print(f"正在下载HumanEval数据集到 {filepath}")
        url = 'https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz'
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"下载完成: {filepath}")
        except Exception as e:
            print(f"下载HumanEval数据集失败: {e}")
            raise
    else:
        print(f"使用现有的HumanEval数据集: {filepath}")
        
    return filepath

def load_humaneval_dataset(filepath: str) -> List[Dict]:
    """
    加载HumanEval数据集
    
    Args:
        filepath: 数据集文件路径
        
    Returns:
        数据集列表
    """
    dataset = []
    
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                item = {
                    'instruction': data['prompt'],
                    'category': data['task_id'],
                    'input': data.get('entry_point', ''),
                    'test': data.get('test', '')
                }
                dataset.append(item)
                
    return dataset

def format_prompt(instruction: str, input_text: str = "", few_shot_examples: List[Dict] = None) -> str:
    """
    
    
    Args:
        instruction: 指令
        input_text: 输入文本
        few_shot_examples: few-shot示例列表
        
    Returns:
        格式化的提示
    """
    # 添加few-shot示例（如果有）
    few_shot_context = ""
    if few_shot_examples:
        for example in few_shot_examples:
            ex_instruction = example.get("instruction", "").strip()
            ex_input = example.get("input", "").strip()
            ex_output = example.get("output", "").strip()
            
            if ex_input:
                few_shot_example = ALPACA_TEMPLATE["prompt_input"].format(
                    instruction=ex_instruction,
                    input=ex_input
                ) + ex_output
            else:
                few_shot_example = ALPACA_TEMPLATE["prompt_no_input"].format(
                    instruction=ex_instruction
                ) + ex_output
                
            few_shot_context += few_shot_example + "\n\n"
    
    # 格式化当前示例的提示
    if input_text:
        prompt = few_shot_context + ALPACA_TEMPLATE["prompt_input"].format(
            instruction=instruction,
            input=input_text
        )
    else:
        prompt = few_shot_context + ALPACA_TEMPLATE["prompt_no_input"].format(
            instruction=instruction
        )
        
    return prompt

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    计算pass@k指标
    
    Args:
        n: 每个问题的总尝试次数
        c: 正确的尝试次数
        k: k值
        
    Returns:
        pass@k值
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    
    return 1.0 - np.prod(1.0 - np.arange(1, c + 1) / (n - np.arange(0, c)))

def eval_code_generation(
    model_path: str,
    base_model: str = "", 
    data_root: str = "./data",
    output_dir: str = "./results",
    num_samples: int = None,
    num_shots: int = 3,
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 40,
    max_new_tokens: int = 512,
    samples_per_problem: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    load_in_8bit: bool = False,
    debug: bool = False
) -> Dict:
    """
    
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始在HumanEval基准上评估模型: {model_path}")
    
    # 加载模型和分词器
    print(f"加载模型和分词器...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        
        # 加载模型
        if load_in_8bit:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                load_in_8bit=True
            )
            # 加载LoRA权重
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model
            ).to(device)
            
            # 如果是LoRA模型，加载LoRA权重
            if model_path.endswith('.ckpt') or model_path.endswith('.pt'):
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, model_path)
                except:
                    # 尝试直接加载模型权重
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise
        
    model.eval()
    
    # 生成文本管道
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 下载并加载HumanEval数据集
    fp = download_humaneval_dataset(data_root)
    dataset = load_humaneval_dataset(fp)
    
    print(f"加载了 {len(dataset)} 个HumanEval问题")
    
    # 限制评估样本数
    if num_samples and num_samples < len(dataset):
        dataset = dataset[:num_samples]
        print(f"仅评估前 {num_samples} 个问题")
    
    # 选择few-shot示例（如果需要）
    few_shot_examples = []
    if num_shots > 0:
        # 加载几个示例问题和参考答案
        # 这些可以从同一个数据集中选择，但不应该与评估样本重叠
        # 或者预定义几个高质量的示例
        # 这里只是简单地选择一些示例
        example_indices = list(range(len(dataset), len(dataset) + num_shots))
        for i in example_indices:
            if i < len(dataset):
                few_shot_examples.append({
                    "instruction": dataset[i]["instruction"],
                    "input": dataset[i]["input"],
                    "output": "# 这里是参考答案" # 实际应该提供一个高质量的参考答案
                })
    
    # 评估
    answers_file = os.path.join(output_dir, "humaneval_answers.jsonl")
    all_answers = []
    
    for i, problem in enumerate(tqdm(dataset)):
        task_id = problem["category"]
        instruction = problem["instruction"]
        input_text = problem.get("input", "")
        
        prompt = format_prompt(instruction, input_text, few_shot_examples)
        
        # 生成多个答案
        answers = []
        for _ in range(samples_per_problem):
            try:
                outputs = generate_text(prompt)
                
                if isinstance(outputs, list) and len(outputs) > 0:
                    generated_text = outputs[0]['generated_text']
                else:
                    generated_text = outputs['generated_text']
                
                # 清理代码
                cleaned_code = clean_code(generated_text)
                answers.append(cleaned_code)
                
                if debug:
                    print(f"\n生成的代码 ({task_id}):")
                    print(cleaned_code)
                    print("-" * 40)
                
            except Exception as e:
                print(f"生成失败 ({task_id}): {e}")
                answers.append("")  # 添加空答案
        
        # 记录结果
        for answer in answers:
            all_answers.append({"task_id": task_id, "completion": answer})
    
    # 保存生成的答案
    with open(answers_file, 'w') as f:
        for answer in all_answers:
            f.write(json.dumps(answer) + '\n')
    
    print(f"已保存所有生成的答案到: {answers_file}")
    
    # 尝试运行官方评估工具
    try:
        # 检查是否安装了human_eval包
        import importlib.util
        if importlib.util.find_spec("human_eval") is not None:
            print("检测到human_eval包，尝试运行官方评估...")
            
            # 运行评估
            result = subprocess.run(
                ["evaluate_functional_correctness", answers_file],
                capture_output=True,
                text=True
            )
            
            print("评估结果:")
            print(result.stdout)
            
            # 解析结果
            results_match = re.search(r"(\w+): pass@1: ([\d\.]+), pass@10: ([\d\.]+)", result.stdout)
            if results_match:
                pass_at_1 = float(results_match.group(2))
                pass_at_10 = float(results_match.group(3))
                
                pass_at_k_results = {
                    "pass@1": pass_at_1,
                    "pass@10": pass_at_10
                }
                
                # 保存结果
                with open(os.path.join(output_dir, "humaneval_results.json"), 'w') as f:
                    json.dump(pass_at_k_results, f, indent=2)
                
                return pass_at_k_results
        else:
            print("未检测到human_eval包，无法运行官方评估")
            print("请安装human_eval包并手动运行评估:")
            print(f"pip install -e git+https://github.com/openai/human-eval.git#egg=human_eval")
            print(f"evaluate_functional_correctness {answers_file}")
    except Exception as e:
        print(f"尝试运行评估工具时出错: {e}")
    
    return {"status": "generated", "answers_file": answers_file}

if __name__ == "__main__":
    # 示例用法
    results = eval_code_generation(
        model_path="",
        base_model="",
        data_root="./data",
        output_dir="./results",
        num_samples=20,  # 设为None以评估全部问题
        num_shots=3,
        temperature=0.1,
        samples_per_problem=5,
        debug=True
    )
    
    print(f"最终结果: {results}")