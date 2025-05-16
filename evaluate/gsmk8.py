import torch
import re
from transformers import pipeline, AutoTokenizer,  AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import time
import os
#transfrmer = 3.40
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 选择 GPU

def extract_answer(text: str) -> int:
    """
    在文本中优先检索以下两种形式之一（只返回匹配到的第一个数字）：
      1) 一个或多个 '#' 后面紧跟空格（可选）再紧跟数字: e.g. ###18
      2) "The answer is " 后面紧跟数字（可选尾随句号）: e.g. The answer is 60.
    如果在文本中找不到匹配，则返回 None。
    """
    # pattern 分别用两个捕获组：group(1) 匹配 ###18 这种形式，group(2) 匹配 The answer is 60. 这种形式
    pattern = re.compile(r"(#+\s*(\d+))|(?:The answer is\s*(\d+)\.?)")
    match = pattern.search(text)
    if match:
        # 若是匹配到 “#+ 数字”，则数字在 group(2)
        # 若是匹配到 “The answer is 数字”，则数字在 group(3)
        # 注意这里 group(1) 捕获了整段 “###18”，我们真正想要的数字部分是 group(2)
        # 而 "The answer is" 的数字在 group(3)
        number_str = match.group(2) or match.group(3)
        return int(number_str) if number_str else None
    return None

def test_llama3_base_on_gsm8k(num_samples=10):
    device = 0 if torch.cuda.is_available() else -1
    
    # 从基础模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # 加载微调后的模型
    model = AutoModelForCausalLM.from_pretrained(
        "",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 创建pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # 加载 GSM8K 数据集
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    correct_count = 0
    wrong_questions = []  # 存储错误的题号

    with open("improved_results.txt", "w", encoding="utf-8") as f:
        f.write("GSM8K测试结果\n")
        f.write("=" * 50 + "\n")

    # 改进的提示模板
    prompt_template = """You are an expert at solving math word problems using careful step-by-step reasoning.

For each problem:
1. Understand what the problem is asking
2. Identify the variables and equations
3. Solve step by step, showing all calculations 
4. Double-check your answer
5. Format your final answer as ###[number] (just the number with no units)

Example 1:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: I need to find the total number of clips Natalia sold in April and May.

In April: 48 clips
In May: 48 ÷ 2 = 24 clips (half as many as April)

Total clips = April clips + May clips
Total clips = 48 + 24 = 72
###72
Example 2:
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: I need to find how much Weng earned for 50 minutes of babysitting.

Hourly rate: $12
Rate per minute: $12 ÷ 60 = $0.2 per minute
Earnings for 50 minutes: 50 * $0.2 = $10

###10
Question: {question}
Answer:"""

    for i, sample in enumerate(test_data):
        question = sample["question"]
        correct_answer = extract_answer(sample["answer"])

        print(f"\n📌 问题 {i+1}: {question}")
        
        prompt = prompt_template.format(question=question)

        # 调整模型生成参数
        response = pipe(
            prompt, 
            max_new_tokens=700,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )
        model_output = response[0]['generated_text']

        # 提取答案
        model_answer = extract_answer(model_output)

        print(f"✅ 真实答案: {correct_answer}")
        print(f"🤖 生成的答案: {model_answer}")

        # 计算正确率
        if model_answer is not None and correct_answer is not None and np.isclose(model_answer, correct_answer, atol=1e-2):
            correct_count += 1
        else:
            wrong_questions.append(i + 1)  # 记录题号

            # 将错误题目写入 TXT 文件
            with open("improved_wrong_answers.txt", "a", encoding="utf-8") as f:
                f.write(f"📌 题号: {i+1}\n")
                f.write(f"📖 题目: {question}\n")
                f.write(f"✅ 正确答案: {correct_answer}\n")
                f.write(f"❌ 生成的答案: {model_answer}\n")
                f.write(f"生成内容: {model_output}\n")
                f.write(f"正确率: {correct_count / (i + 1) * 100:.2f}%\n")
                f.write("-" * 50 + "\n")

        # 计算并输出当前正确率
        current_accuracy = (correct_count / (i + 1)) * 100
        print(f"📊 当前正确率: {current_accuracy:.2f}%（{correct_count}/{i+1}）")

    # 总结错误题目
    if wrong_questions:
        print("\n📌 错误的题目编号:", wrong_questions)
        print("📂 已将错误题目保存到 `improved_wrong_answers.txt`")
    
    print(f"最终准确率: {(correct_count / len(test_data)) * 100:.2f}%")
    

if __name__ == "__main__":
    start = time.time()
    test_llama3_base_on_gsm8k(num_samples=10)
    print(f"Total time: {time.time() - start:.2f}s")



# 


