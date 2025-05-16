# few-shot evaluation for GPT-2 on MMLU

import torch
import transformers
from datasets import load_dataset
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import random
import json
import re
from peft import LoraConfig
from peft import PeftModel, get_peft_model
from huggingface_hub import login

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def format_example(instruction, input_text, output):
    """Format a single example using a simple direct format"""
    if input_text:
        return f"{instruction}\n{input_text}\n{output}"
    else:
        return f"{instruction}\n{output}"

def extract_option(text):
    """Extract the option letter (A, B, C, or D) from a response"""
    # Common patterns in responses
    patterns = [
        r"[Tt]he answer is:?\s*([A-D])[.\s]",  # "The answer is: A." or "The answer is A"
        r"[Aa]nswer:?\s*([A-D])[.\s]",         # "Answer: B." or "Answer B"
        r"[Oo]ption:?\s*([A-D])[.\s]",         # "Option: C." or "Option C"
        r"[Cc]hoice:?\s*([A-D])[.\s]",         # "Choice: D." or "Choice D"
        r"^([A-D])[.\s]",                       # Just starts with "A." or "A "
        r"\s([A-D])$",                          # Just ends with " A"
        r"\s([A-D])[.\s]",                      # Contains " A." or " A "
        r"([A-D])"                              # Just contains A, B, C, or D as a fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def load_json_dataset(file_path):
    """Load dataset from a JSON file (array of objects)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, but got {type(data).__name__}")
            
        print(f"Successfully loaded {len(data)} examples from {file_path}")
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        raise

def evaluate_on_custom_dataset(
        model,
        tokenizer,
        dataset_path='',
        num_samples=100,
        temperature=0.0,
        max_new_tokens=128,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    Evaluate model performance on a custom dataset using 2-shot approach with fixed examples
    """
    nltk.download('punkt', quiet=True)
    
    print("Loading evaluation metrics")
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    model.to(device)
    model.eval()
        
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        return_full_text=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1
    )

    print(f"Loading dataset from {dataset_path}")
    dataset = load_json_dataset(dataset_path)
    
    # Limit the number of samples if needed
    if num_samples and num_samples < len(dataset):
        eval_samples = dataset[:num_samples]
    else:
        eval_samples = dataset
        num_samples = len(eval_samples)
    
    # Define the fixed 2-shot examples
    shot_examples = [
        {
            "instruction": "Please answer the following multiple-choice question honestly.",
            "input": "Qusetion: Which of the following is NOT true of executive orders? Options: A. Presidents avoid using executive orders for controversial actions., B. Executive orders have the same effect as laws passed by Congress., C. Presidents have made increased use of executive orders since the 1970s., D. Executive orders bypass congressional approval.",
            "output": "The answer is: A. Presidents avoid using executive orders for controversial actions.",
            "class": "high_school_government_and_politics"
        },
        {
            "instruction": "Please answer the following multiple-choice question honestly.",
            "input": "Qusetion: What is direct diplomacy? Options: A. Members of Congress negotiating directly with foreign governments, B. Face-to-face meetings between state leaders, C. The president consulting Congress on foreign policy issues, D. Bilateral talks that do not involve a third-party negotiator",
            "output": "The answer is: A. Members of Congress negotiating directly with foreign governments",
            "class": "us_foreign_policy"
        }
    ]
    
    print(f"Using {len(shot_examples)} fixed shot examples")
    print(f"Evaluation sample count: {len(eval_samples)}")

    all_predictions = []
    all_references = []
    bleu_scores = []
    correct_options = 0
    total_questions = 0

    smoothie = SmoothingFunction().method1
    
    for i, example in enumerate(eval_samples):
        instruction = example["instruction"].strip()
        input_text = example["input"].strip() if example["input"] else ""
        reference = example["output"].strip()
        
        # Build 2-shot context with the fixed examples
        few_shot_context = ""
        for shot in shot_examples:
            shot_instruction = shot["instruction"].strip()
            shot_input = shot["input"].strip() if shot["input"] else ""
            shot_output = shot["output"].strip()
            
            few_shot_example = format_example(shot_instruction, shot_input, shot_output)
            few_shot_context += few_shot_example + "\n\n"
        
        # Add prompt for the current evaluation sample
        if input_text:
            prompt = few_shot_context + f"{instruction}\n{input_text}\n"
        else:
            prompt = few_shot_context + f"{instruction}\n"

        try:
            outputs = generate_text(prompt)
            prediction = outputs[0]['generated_text'].strip()
        except Exception as e:
            print(f"Generation failed ({i+1}/{num_samples}): {e}")
            prediction = ""

        reference_tokens = nltk.word_tokenize(reference.lower())
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        bleu_score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

        # Extract and compare the options
        pred_option = extract_option(prediction)
        ref_option = extract_option(reference)
        
        total_questions += 1
        if pred_option and ref_option and pred_option == ref_option:
            correct_options += 1
            option_match = "✓"
        else:
            option_match = "✗"

        all_predictions.append(prediction)
        all_references.append(reference)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed samples: {i+1}/{num_samples}, current avg BLEU: {sum(bleu_scores) / (i+1):.4f}")
            print(f"Example {i}:")
            print(f"  Instruction: {instruction[:100]}...")
            print(f"  Input: {input_text[:100]}...")
            print(f"  Prediction: {prediction[:100]}... [Option: {pred_option or 'Not found'}]")
            print(f"  Reference: {reference[:100]}... [Option: {ref_option or 'Not found'}]")
            print(f"  Option Match: {option_match}, BLEU score: {bleu_score:.4f}")
            print("-" * 40)

    meteor_result = meteor.compute(predictions=all_predictions, references=all_references)
    rouge_result = rouge.compute(predictions=all_predictions, references=all_references, use_stemmer=True)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    # Calculate accuracy for multiple-choice questions
    accuracy = correct_options / total_questions if total_questions > 0 else 0

    metrics = {
        "Accuracy": accuracy,
        "BLEU": avg_bleu,
        "METEOR": meteor_result['meteor'],
        "ROUGE-L": rouge_result['rougeL'],
        "correct_count": correct_options,
        "total_count": total_questions
    }

    print("\nEvaluation Results:")
    print(f"Multiple-Choice Accuracy: {metrics['Accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_count']})")
    for metric_name, score in metrics.items():
        if metric_name not in ["Accuracy", "correct_count", "total_count"]:  # Already printed above
            print(f"{metric_name}: {score:.4f}")

    # Print the fixed few-shot examples used
    print("\n2-Shot Examples Used:")
    for i, shot in enumerate(shot_examples):
        print(f"Example {i+1}:")
        print(f"  Instruction: {shot['instruction']}")
        if shot['input']:
            print(f"  Input: {shot['input']}")
        print(f"  Output: {shot['output']}")
        if 'class' in shot:
            print(f"  Class: {shot['class']}")
        print("-" * 40)

    return metrics

if __name__ == "__main__":
    # Huggingface login is optional for GPT-2 as it's a publicly available model
    # login(token="your_token_here")  # Only needed for protected models
    
    # Load the GPT-2 model
    model_name = 'Qwen/Qwen2.5-7B-Instruct'  # You can use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' for larger versions
    print(f"Loading Qwen-2 model: {model_name}")
    
    # Load GPT-2 model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        '',
        device_map="auto"
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )
    
    # GPT-2's tokenizer doesn't have a pad_token set by default
    # We'll use the EOS token as the pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # If you want to evaluate with LoRA weights, uncomment and modify these lines
    # peft_config = LoraConfig(
    #     peft_type="LORA",
    #     r=8, 
    #     lora_alpha=32,
    #     target_modules=["c_attn", "c_proj"]  # GPT-2 specific target modules
    # )
    # peft_model = get_peft_model(model, peft_config)
    # checkpoint_path = '/path/to/your/lora/checkpoint.ckpt'
    # state_dict = torch.load(checkpoint_path, map_location="cpu")
    # peft_model.load_state_dict(state_dict, strict=False)
    # model = peft_model

    print("Starting evaluation...")
    metrics = evaluate_on_custom_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset_path='',
        num_samples=500,  # Reduced from 1400 for initial testing with GPT-2
        temperature=0.7,  # GPT-2 often works better with slightly higher temperature
        max_new_tokens=128
    )

    print("Evaluation completed!")
    print(f"Multiple-Choice Accuracy: {metrics['Accuracy']:.4f}")
    print(f"BLEU: {metrics['BLEU']:.4f}")
    print(f"METEOR: {metrics['METEOR']:.4f}")
    print(f"ROUGE-L: {metrics['ROUGE-L']:.4f}")
    
    # Save results to file
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    file_path = f"{result_dir}/qwen_mmlu_evaluation.txt"
    
    with open(file_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Type: 2-shot with fixed examples\n")
        f.write(f"Multiple-Choice Accuracy: {metrics['Accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_count']})\n")
        f.write(f"BLEU: {metrics['BLEU']:.4f}\n")
        f.write(f"METEOR: {metrics['METEOR']:.4f}\n")
        f.write(f"ROUGE-L: {metrics['ROUGE-L']:.4f}\n")