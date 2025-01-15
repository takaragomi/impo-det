import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
from models import model_selection


def _load_data(data_path):
    df = pd.read_csv(data_path)
    df['context'] = df.groupby('context_label')['user_comment'].transform(lambda x: ' '.join(x))
    return df

data_path = 'data/sample.csv'
df = _load_data(data_path)
model_name = "elyza/Llama-3-ELYZA-JP-8B"
model, tokenizer = model_selection.load_model(model_name)

# モデルのセットアップ
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。与えられた文脈と発言を元に、それが嘘か本当かを判断してください。"

# 推論関数
def predict_label(context, user_comment):
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"文脈: {context}\n発言: {user_comment}\nこの発言は嘘ですか？TRUE または FALSE で答えてください。"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=50,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    output = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
    )
    return output.strip()

predicted_labels = []
for i, row in df.iterrows():
    context = row['context']
    user_comment = row['user_comment']
    prediction = predict_label(context, user_comment)
    print(prediction)
    # 文字列を対応するbool型に変換
    predicted_labels.append(True if prediction == "TRUE" else False)

df['pred_label'] = predicted_labels

# 正解率の算出
accuracy = accuracy_score(df['label'], df['pred_label'])
print(f"正解率: {accuracy * 100:.2f}%")

# 結果をJSONで保存
output_data = df.to_dict(orient='records')

with open("prediction_results.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# main関数のテンプレート
# def main():
#     # データの選択
#     input_path = Path("")
#     model = #モデルの選択
#     prompt_type =   # プロンプトの選択
#     output_path = # 実験結果のパス
#     # 実行スクリプト

# if __name__ == "__main__":
#     main()