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
import time
from tqdm import tqdm
import logging
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
random.seed(12345)

# Configuration
model_path = "meta-llama/Llama-2-7b-hf"
output_dir = "results"
num_shots = 5  # Number of few-shot examples
max_new_tokens = 128
temperature = 0.1
num_test_examples = 1000  # Number of test examples to evaluate

def load_mmlu_pro():
    """Load the MMLU-Pro dataset from HuggingFace datasets"""
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    
    # Preprocess to filter out N/A options
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    
    return test_df, val_df

def preprocess(dataset):
    """Filter out N/A options"""
    result = []
    for item in dataset:
        options = []
        for opt in item["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        
        # Make a copy to avoid modifying the original dataset
        processed_item = item.copy()
        processed_item["options"] = options
        result.append(processed_item)
    
    return result

def select_by_category(df, subject):
    """Filter dataset by subject/category"""
    return [item for item in df if item["category"] == subject]

def format_example(example, including_answer=True):
    """Format a question example with step-by-step reasoning"""
    prompt = "Question: "
    question = example["question"]
    options = example["options"]
    
    prompt += question + "\n"
    prompt += "Options:\n"
    
    for i, opt in enumerate(options):
        prompt += f"{choices[i]}. {opt}\n"
    
    if including_answer:
        # Get answer letter from answer_index
        answer = choices[example["answer_index"]]
        prompt += f"Answer: Let's think step by step.\n"
        # Add reasoning if available
        if "reasoning" in example and example["reasoning"]:
            prompt += example["reasoning"] + "\n"
        prompt += f"The answer is {answer}.\n\n"
    else:
        prompt += "Answer: Let's think step by step.\n"
    
    return prompt

def generate_prompt(val_examples, test_example):
    """Generate few-shot prompt with validation examples"""
    prompt = "Answer the following multiple-choice questions by reasoning step by step.\n\n"
    
    # Add examples with answers from same category as test example
    category = test_example["category"]
    category_examples = [ex for ex in val_examples if ex["category"] == category]
    
    # Use up to num_shots examples
    for example in category_examples[:num_shots]:
        prompt += format_example(example, including_answer=True)
    
    # Add test example without answer
    prompt += format_example(test_example, including_answer=False)
    
    return prompt

def extract_answer(text):
    """Extract answer option from model output"""
    # Pattern to match 'the answer is X' with variations
    patterns = [
        r"[Tt]he answer is:?\s*([A-P])[.\s]",
        r"[Aa]nswer:?\s*([A-P])[.\s]",
        r"[Tt]he correct answer is:?\s*([A-P])[.\s]",
        r"[Tt]herefore,? the answer is:?\s*([A-P])[.\s]",
        r"[Ss]o the answer is:?\s*([A-P])[.\s]",
        r"[Tt]hus,? the answer is:?\s*([A-P])[.\s]",
        r"[Ff]inal answer:?\s*([A-P])[.\s]",
        r"\s([A-P])\s*is the (correct |right |)answer",
        r"^\s*([A-P])[.\s]",  # Just starts with option letter
        r"\s([A-P])$",        # Just ends with option letter
        r"\s([A-P])[.\s]",    # Contains option letter
        r"([A-P])"            # Any option letter as fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def evaluate_mmlu_pro(model, tokenizer):
    """Evaluate model on MMLU-Pro dataset"""
    nltk.download('punkt', quiet=True)
    
    logging.info("Loading evaluation metrics")
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bleu_smoothing = SmoothingFunction().method1
    
    # Load dataset
    logging.info("Loading MMLU-Pro dataset")
    test_data, val_data = load_mmlu_pro()
    
    # Randomly select examples for testing
    if len(test_data) > num_test_examples:
        logging.info(f"Randomly selecting {num_test_examples} examples from {len(test_data)} total")
        test_data = random.sample(test_data, num_test_examples)
    
    # Generate text configuration
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        device_map="auto",
        torch_dtype=torch.bfloat16,
        return_full_text=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1
    )
    
    # Stats counters
    correct_count = 0
    total_count = 0
    all_predictions = []
    all_references = []
    bleu_scores = []
    
    # Evaluate with progress bar
    logging.info(f"Starting evaluation with {num_shots}-shot prompting on {len(test_data)} examples")
    for example in tqdm(test_data, desc="Evaluating"):
        # Generate few-shot prompt
        prompt = generate_prompt(val_data, example)
        
        # Generate prediction
        try:
            output = generate_text(prompt)
            prediction = output[0]['generated_text'].strip()
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            prediction = ""
        
        # Extract predicted answer letter
        pred_option = extract_answer(prediction)
        
        # Get reference answer letter
        ref_option = choices[example["answer_index"]]
        
        # Format reference answer (for text metrics)
        reference = f"The answer is {ref_option}."
        
        # Check if prediction is correct
        is_correct = pred_option == ref_option
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Calculate BLEU score
        reference_tokens = nltk.word_tokenize(reference.lower())
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        bleu_score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=bleu_smoothing)
        bleu_scores.append(bleu_score)
        
        # Store for ROUGE and METEOR calculation
        all_predictions.append(prediction)
        all_references.append(reference)
        
        # Log some examples
        if total_count % 50 == 0:
            logging.info(f"Example {total_count}:")
            logging.info(f"  Question: {example['question'][:100]}...")
            logging.info(f"  Prediction: {prediction[:100]}... [Option: {pred_option or 'Not found'}]")
            logging.info(f"  Reference: {reference} [Option: {ref_option}]")
            logging.info(f"  Correct: {is_correct}, BLEU score: {bleu_score:.4f}")
            logging.info(f"  Current accuracy: {correct_count/total_count:.4f} ({correct_count}/{total_count})")
            logging.info("-" * 40)
    
    # Calculate final metrics
    accuracy = correct_count / total_count if total_count > 0 else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    meteor_result = meteor.compute(predictions=all_predictions, references=all_references)
    rouge_result = rouge.compute(predictions=all_predictions, references=all_references, use_stemmer=True)
    
    # Compile results
    metrics = {
        "Accuracy": accuracy,
        "BLEU": avg_bleu,
        "METEOR": meteor_result['meteor'],
        "ROUGE-L": rouge_result['rougeL'],
        "correct_count": correct_count,
        "total_count": total_count
    }
    
    return metrics

def main():
    """Main evaluation function"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logging.info(f"Loading model: {model_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        '',
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    # Start evaluation
    logging.info("Starting MMLU-Pro evaluation")
    metrics = evaluate_mmlu_pro(model, tokenizer)
    
    # Print metrics directly
    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"BLEU: {metrics['BLEU']:.4f}")
    print(f"METEOR: {metrics['METEOR']:.4f}")
    print(f"ROUGE-L: {metrics['ROUGE-L']:.4f}")
    print(f"Correct: {metrics['correct_count']}/{metrics['total_count']}")
    print("="*50 + "\n")
    
    # Log results
    logging.info("\nEvaluation Results:")
    logging.info(f"Accuracy: {metrics['Accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_count']})")
    logging.info(f"BLEU: {metrics['BLEU']:.4f}")
    logging.info(f"METEOR: {metrics['METEOR']:.4f}")
    logging.info(f"ROUGE-L: {metrics['ROUGE-L']:.4f}")
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = model_path.split("/")[-1]
    results_file = os.path.join(output_dir, f"{model_name}_{timestamp}_results.json")
    
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Results saved to {results_file}")
    return metrics

if __name__ == "__main__":
    main()