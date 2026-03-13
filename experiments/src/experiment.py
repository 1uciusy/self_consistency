import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from experiments.utils.evaluation import extract_final_answer, majority_vote, check_answer_correctness


class DeepSeekR1Experiment:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        
        self.model_name = model_name
        
        print(f"Loading DeepSeek R1 model: {model_name}")
        
        # Check for MPS (Metal Performance Shaders) on MacBook
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Using MPS device for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device for acceleration")
        else:
            self.device = torch.device("cpu")
            print(f"Using CPU device")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # For MPS device, use float32 instead of bfloat16 for better numerical stability
        dtype = torch.float32 if self.device.type == "mps" else torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        
        # Move model to the appropriate device
        self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def build_deepseek_prompt(self, question: str) -> str:
        system_prompt = "You are a mathematical reasoning expert. Solve these math word problems step by step. Provide your final answer in the format: #### [number]"
        
        few_shot_text = ""
        
        full_prompt = f"{system_prompt}\n\n{few_shot_text}Question: {question}\nLet's solve this step by step:\n"
        
        return full_prompt
    
    def generate_reasoning_paths(self, 
                               question: str, 
                               num_samples: int = 5, 
                               temperature: float = 0.7, 
                               max_length: int = 1024,
                               top_p: float = 0.95,
                               top_k: int = 50) -> List[str]:
        
        prompt = self.build_deepseek_prompt(question)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        
        paths = []

        for i in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            
            generated_text = self.tokenizer.decode(outputs[0].to(torch.int32).cpu().numpy(), skip_special_tokens=True)
                
            if "Let's solve this step by step:" in generated_text:
                generated_text = generated_text.split("Let's solve this step by step:")[-1].strip()
                
            paths.append(generated_text.strip())

        return paths
    
    def evaluate_question(self, 
                         question: str, 
                         correct_answer: str,
                         num_samples: int = 5,
                         temperature: float = 0.7,
                         max_length: int = 1024,
                         top_p: float = 0.95,
                         top_k: int = 50) -> Dict:
        standard_path = self.generate_reasoning_paths(
            question, num_samples=1, temperature=temperature, max_length=max_length, top_p=top_p, top_k=top_k)[0]
        standard_answer = extract_final_answer(standard_path)
        
        paths = self.generate_reasoning_paths(question, num_samples=num_samples, temperature=temperature, max_length=max_length, top_p=top_p, top_k=top_k)
        for i, path in enumerate(paths):
            print(f"Path {i+1}: {path}")
        answers = [extract_final_answer(path) for path in paths]
        sc_answer = majority_vote(answers)
        
        standard_correct = check_answer_correctness(standard_answer, correct_answer)
        sc_correct = check_answer_correctness(sc_answer, correct_answer)
        
        return {
            'standard_correct': standard_correct,
            'sc_correct': sc_correct,
            'standard_path': standard_path,
            'sc_paths': paths,
            'standard_answer': standard_answer,
            'sc_answer': sc_answer
        }