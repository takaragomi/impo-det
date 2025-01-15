import csv

# CSVファイルを読み込む
input_file = 'data/sample.csv'  # 入力CSVファイル名
output_file = 'output.csv'  # 出力CSVファイル名

# 編集後の行を格納するリスト
edited_rows = []

# CSVファイルを開いて読み込み
with open(input_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # user_id と user_comment を結合して新しい形式にする
        row['user_comment'] = f"{row['user_id']}:{row['user_comment']}"
        edited_rows.append(row)

# 編集結果を新しいCSVファイルに書き込む
with open(output_file, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(edited_rows)

print(f"編集されたデータは {output_file} に保存されました。")