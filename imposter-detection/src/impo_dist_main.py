import json
from pathlib import Path
from typing import Dict, List, Callable
import pandas as pd
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
from models import model_selection
from enum import Enum, auto
from prompts.imposter_detection_prompts import get_default_system_prompt, get_zero_shot_dist_prompt, get_one_shot_dist_prompt, get_few_shot_dist_prompt
from datetime import datetime
from pathlib import Path
import csv

class PromptType(Enum):
    ZERO_SHOT_DIST = auto()
    ONE_SHOT_DIST = auto()
    FEW_SHOT_DIST = auto()
    # 必要に応じて他のプロンプトタイプを追加

PROMPT_FUNC_MAPPING = {
    PromptType.ZERO_SHOT_DIST: get_zero_shot_dist_prompt,
    PromptType.ONE_SHOT_DIST: get_one_shot_dist_prompt,
    PromptType.FEW_SHOT_DIST: get_few_shot_dist_prompt
}

def get_output_path(prompt_type: PromptType, input_path: Path) -> Path:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name = input_path.stem
    output_dir = Path("outputs") / prompt_type.name.lower()  # "outputs/zero_shot"など
    output_path = output_dir / f"{current_time}_{data_name}.csv" #json #csv
    return output_path

class ImposterDetectProcessor:
    def __init__(self, model_name: str, 
        system_prompt: str = None, 
        prompt_func: Callable[[str, str], str] = None):
        self.model, self.tokenizer = model_selection.load_model(model_name)
        self.system_prompt = system_prompt if system_prompt is not None else get_default_system_prompt()
        self.prompt_func = prompt_func if prompt_func is not None else get_zero_shot_dist_prompt

    def process_file(self, input_path: Path, output_path: Path):
        try:
            df = self._load_data(input_path)
            output_impo_data = self._get_results(df)
            # output_psyc_data = self._get_results(df)
            output_path = self._save_output(output_impo_data, output_path)
            print(f'{output_path}に保存されました')
            
        except Exception as e:
            print("エラーが発生しました:")
            print(f"エラーの詳細: {e}")
            traceback.print_exc()
    
    def _load_data(self, input_path):
        df = pd.read_csv(input_path)
        df['context'] = df.groupby('context_label')['user_comment'].transform(lambda x: ' '.join(x))
        return df
    
    def _get_prompt(self, context, user_comment): 
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_func(context, user_comment)},
        ]
        # print(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        return token_ids
    
    def _predict_label(self, context, user_comment):
        token_ids = self._get_prompt(context, user_comment)
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=50,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
        output = self.tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
        )
        return output.strip()
    
    def _get_results(self, df):
        distincted_impo_labels = []
        #df = df[df['psychological_label'] != 0]
        df = df[df['camp_label'] >= 0]
        for i, row in df.iterrows():
            context = row['context']
            user_comment = row['user_comment']
            prediction = self._predict_label(context, user_comment)
            print(prediction)
            import re
            match = re.search(r'\d', prediction)  # Find the first digit in the string
            if match:
                distincted_impo_labels.append(int(match.group()))
            else:
                distincted_impo_labels.append(None)
            # predicted_psyc_labels.append(int(' '.join([char for char in prediction if char.isdigit()])))
         
        df['dist_impo_label'] = distincted_impo_labels
        print(distincted_impo_labels)

        impo_accuracy = accuracy_score(df['camp_label'], df['dist_impo_label'])
        impo_accuracy = f"陣営判別 正解率: {impo_accuracy * 100:.2f}%"
        df['impo_accuracy'] = impo_accuracy
        output_impo_data = df.to_dict(orient='records')

        return output_impo_data
    
    def _save_output_to_json(self, output_impo_data, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_impo_data, f, ensure_ascii=False, indent=4)
        return output_path

    def _save_output(self, output_impo_data, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            if isinstance(output_impo_data, list) and all(isinstance(item, dict) for item in output_impo_data):
                writer = csv.DictWriter(f, fieldnames=output_impo_data[0].keys())
                writer.writeheader()
                writer.writerows(output_impo_data)
            else:
                raise ValueError("Output data must be a list of dictionaries to save as CSV.")
        return output_path

def main():
    model_name = "elyza/Llama-3-ELYZA-JP-8B"
    system_prompt = get_default_system_prompt()

    prompt_type = PromptType.FEW_SHOT_DIST #ZERO_SHOT #ONE_SHOT #FEW_SHOT #ZERO_SHOT_DIST #ONE_SHOT_DIST #FEW_SHOT_DIST
    prompt_func = PROMPT_FUNC_MAPPING[prompt_type]

    input_path = Path("data/sample.csv")
    output_path = get_output_path(prompt_type, input_path)

    processor = ImposterDetectProcessor(
        model_name,
        system_prompt=system_prompt,
        prompt_func=prompt_func
    )
    
    processor.process_file(input_path, output_path)

if __name__ == "__main__":
    main()