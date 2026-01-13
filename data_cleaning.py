# import json
# import string

# def process_description(description):
#     # 1. 移除所有标点符号（方法2的核心步骤）
#     translator = str.maketrans('', '', string.punctuation)
#     cleaned = description.translate(translator)
    
#     # 2. 按空白拆分单词（自动处理连续空格）
#     words = cleaned.split()
    
#     # 3. 截断到80词（如果超过）
#     if len(words) > 80:
#         truncated_words = words[:80]
#         # 拼接回字符串（用空格分隔）
#         return ' '.join(truncated_words)
#     else:
#         # 未超过80词时，返回原始描述（保留原始标点）
#         return description

# # 处理文件
# input_path = 'image_descriptions.jsonl'
# output_path = 'cleaned_image_descriptions.jsonl'

# with open(input_path, 'r', encoding='utf-8') as infile, \
#      open(output_path, 'w', encoding='utf-8') as outfile:
    
#     for line in infile:
#         # 解析JSON行
#         data = json.loads(line.strip())
#         # 处理描述字段
#         if 'description' in data:
#             data['description'] = process_description(data['description'])
#         # 写入新文件
#         outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# print(f"处理完成，结果已保存到 {output_path}")


import json
from tqdm import tqdm

input_path = "processed_results_with_subtext.jsonl"
output_path = "processed_results_with_subtext_clean_strict.jsonl"

total = 0
kept = 0
removed = 0
removed_samples = []

# === 第一步：先统计 subtext 字段的所有可能键 ===
all_subtext_keys = set()
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            subtexts = item.get("subtext", [])
            for sub in subtexts:
                if isinstance(sub, dict):
                    all_subtext_keys.update(sub.keys())
        except Exception:
            continue

print(f"【发现的 subtext 字段全集】共 {len(all_subtext_keys)} 个字段：\n{sorted(all_subtext_keys)}\n")

# === 第二步：开始清洗 ===
with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="清洗进度"):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            removed += 1
            removed_samples.append(f"行{total}：JSON解析错误")
            continue

        subtexts = item.get("subtext", [])
        if not isinstance(subtexts, list) or len(subtexts) == 0:
            removed += 1
            removed_samples.append(item.get("image_name", f"行{total}：subtext缺失"))
            continue

        # 判断是否每个 subtext 都有全部键
        valid = True
        for sub in subtexts:
            if not isinstance(sub, dict):
                valid = False
                break
            missing_keys = all_subtext_keys - sub.keys()
            if missing_keys:
                valid = False
                break

        if not valid:
            removed += 1
            removed_samples.append(item.get("image_name", f"行{total}"))
            continue

        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        kept += 1

print("\n=== 严格数据清洗完成 ===")
print(f"总样本数: {total}")
print(f"保留样本: {kept}")
print(f"删除样本: {removed}")
print("\n被删除的样本:")
for name in removed_samples[:30]:  # 只打印前30个避免刷屏
    print(" -", name)
if len(removed_samples) > 30:
    print(f"... 共 {len(removed_samples)} 条被删除。")

print(f"\n清洗后的文件已保存到: {output_path}")
