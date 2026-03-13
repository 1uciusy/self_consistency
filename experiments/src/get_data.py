from pathlib import Path
import json
from datasets import load_dataset


def download_gsm8k(save_dir: str = None):
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "data"
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"Downloading GSM8K dataset...")
    
    dataset = load_dataset("gsm8k", "main")
    
    for split in ["train", "test"]:
        data = dataset[split]
        save_path = save_dir / f"gsm8k_{split}.json"
        
        formatted_data = []
        for item in data:
            formatted_item = {
                "question": item["question"],
                "answer": item["answer"]
            }
            formatted_data.append(formatted_item)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {split} split to {save_path} ({len(formatted_data)} items)")
    
    print("GSM8K dataset downloaded successfully!")


if __name__ == "__main__":
    download_gsm8k()
