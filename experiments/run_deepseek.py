import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from experiments.src.experiment import DeepSeekR1Experiment


def load_gsm8k_data(split: str = "test"):
    data_path = Path(__file__).parent / "data" / f"gsm8k_{split}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text


def run_deepseek_experiment(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    num_samples: int = 5,
    temperature: float = 0.7,
    max_length: int = 512,
    top_p: float = 0.95,
    top_k: int = 50,
    max_questions: int = None
):
    print("=" * 60)
    print("DeepSeek R1 Self-Consistency Experiment")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Num samples: {num_samples}")
    print(f"Temperature: {temperature}")
    print(f"Max length: {max_length}")
    print()
    
    experiment = DeepSeekR1Experiment(model_name=model_name)
    
    test_data = load_gsm8k_data("test")
    
    if max_questions is not None:
        test_data = test_data[:max_questions]
    
    print(f"\nTesting on {len(test_data)} questions from GSM8K test set\n")
    
    results = []
    standard_correct_count = 0
    sc_correct_count = 0
    
    with tqdm(total=len(test_data), desc="Processing questions") as pbar:
        for i, item in enumerate(test_data):
            question = item["question"]
            correct_answer = extract_gsm8k_answer(item["answer"])
            eval_result = experiment.evaluate_question(
                question=question,
                correct_answer=correct_answer,
                num_samples=num_samples,
                temperature=temperature,
                max_length=max_length,
                top_p=top_p,
                top_k=top_k
            )
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "standard_answer": eval_result["standard_answer"],
                "sc_answer": eval_result["sc_answer"],
                "standard_correct": eval_result["standard_correct"],
                "sc_correct": eval_result["sc_correct"],
                "standard_path": eval_result["standard_path"],
                "sc_paths": eval_result["sc_paths"]
            })
            
            if eval_result["standard_correct"]:
                standard_correct_count += 1
            if eval_result["sc_correct"]:
                sc_correct_count += 1
            
            pbar.update(1)
            
            if (i + 1) % 10 == 0:
                current_std_acc = standard_correct_count / (i + 1)
                current_sc_acc = sc_correct_count / (i + 1)
                print(f"\nProgress: {i+1}/{len(test_data)}")
                print(f"Current Standard Accuracy: {current_std_acc:.2%}")
                print(f"Current Self-Consistency Accuracy: {current_sc_acc:.2%}")
                print()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    standard_accuracy = standard_correct_count / len(test_data)
    sc_accuracy = sc_correct_count / len(test_data)
    
    print(f"Standard Chain-of-Thought Accuracy: {standard_accuracy:.2%}")
    print(f"Self-Consistency Accuracy: {sc_accuracy:.2%}")
    print(f"Improvement: {(sc_accuracy - standard_accuracy):+.2%}")
    
    output_file = Path("deepseek_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "num_samples": num_samples,
            "temperature": temperature,
            "standard_accuracy": standard_accuracy,
            "sc_accuracy": sc_accuracy,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    
    return {
        "standard_accuracy": standard_accuracy,
        "sc_accuracy": sc_accuracy,
        "results": results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DeepSeek R1 self-consistency experiment")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="DeepSeek R1 model name")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples for self-consistency")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Maximum number of questions to test (for quick testing)")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum length of generated sequences")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    
    args = parser.parse_args()
        
    run_deepseek_experiment(
        model_name=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_questions=args.max_questions,
        max_length=args.max_length,
        top_p=args.top_p,
        top_k=args.top_k,
    )
