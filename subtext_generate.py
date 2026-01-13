# import json
# from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from pydantic import BaseModel,Field
# import os
# from typing import TypedDict, Annotated,List
# # 1. 定义文件路径（替换为你的文件实际路径）
# file_path = "processed_results.jsonl"

# # 2. 逐行读取并解析JSON
# data_list = []  # 用于存储所有解析后的JSON对象
# with open(file_path, "r", encoding="utf-8") as f:
#     for line_num, line in enumerate(f, 1):  # 枚举行号，方便定位错误
#         try:
#             # 解析当前行的JSON数据
#             json_obj = json.loads(line.strip())  # strip() 去除行首尾空白（如换行符）
#             data_list.append(json_obj)
#         except json.JSONDecodeError as e:
#             # 捕获解析错误，提示具体行号
#             print(f"解析第 {line_num} 行时出错：{e}")
#             continue

# # 3. 验证加载结果（查看前2条数据的结构）
# print(f"总共加载了 {len(data_list)} 条数据")
# if data_list:
#     print("\n第一条数据的结构：")
#     for key, value in data_list[0].items():
#         print(f"- {key}: {type(value).__name__}")  # 打印键名和对应值的类型

# all_datas = []  
# i = 0
# for item in data_list:
#     # 1. 提取 positive 作为列表第一个元素
#     current_group = [item["positive"]]
#     i+=1
#     # 2. 提取 negatives 中的所有 sentence，按顺序添加到列表
#     for neg in item["negatives"]:
#         current_group.append(neg["sentence"])
    
#     # 3. 验证每组是否是 11 个元素（1个pos + 10个neg），避免数据缺失
    
#     if len(current_group) != 11:
#         print(f"警告：图片 {item['image_name']} 仅生成 {len(current_group)} 个元素（应为11个）")
#     sample = {"image_name": item["image_name"], "group": current_group}
#     # 4. 将当前组添加到最终结果
#     all_datas.append(sample)

# llm = ChatOpenAI(
# model_name="deepseek-chat",
# openai_api_base="https://api.deepseek.com/v1",
# openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
# max_tokens=4096,
# temperature=0.1,
# top_p=0.9
# )

# class SubText(BaseModel):
#     noun: Annotated[str, "名词"]
#     adjectives: Annotated[List[str], "形容词列表"]
#     label: Annotated[int, "标签（0表示负样本，1表示正样本）"]
#     sentence: Annotated[str, "包含名词和形容词的描述语句"]



# sample_str = str(all_datas[0]['group']) 
# parser = JsonOutputParser(pydantic_object=SubText)


# def get_chain():
#     TEMPLATE = """
#     {sample_str}
#     请你找出此文本列表中所有的主体名词及其形容词以如下形式返回给我,不需要重复的，并有以下几条规则。
#     1.如果某个名词没有形容词则不需要将该名词返回给我。
#     2.其中第一条为正文本，由第一条构建的一组名词和形容词的label为1
#     3.后面的所有文本为负文本，当有与第一条名词相同但是形容词不同时需返回，并且label为0
#     4.负文本中与正文本相同的名词形容词组只需构建一次
#     5.名词严格按照一个单词给出而不是名词组
#     6.并将拆分出的名词和形容词构建为一个语法正确的短句,需要表达自然，必要时去掉and
#     7.请严格按照我给出的规则返回json格式，不需要多余的内容
#     {format_struction}
#     示例：[{{"noun":"noun1","adjectives":["adj1"],"label":1,"sentence":"a photo containing a/an/the [adj1] noun."}},{{"noun":"noun2","adjectives":["adj1","adj2"],"label":0,"sentence":"a photo containing a/an/the [adj1],and [adj2] noun."}}]
#     """
#     prompt = PromptTemplate(template=TEMPLATE,input_variables=["sample_str"],partial_variables={"format_struction":parser.get_format_instructions()})
#     chain = prompt | llm | parser
#     return chain
# print(get_chain().invoke({"sample_str":sample_str}))

# batch_size = 10

# for i in range(0,len(all_datas),batch_size):
#     batch = all_datas[i:i+batch_size]
#     samples = [{"sample_str":str(sample['group'])} for sample in batch]
#     chain = get_chain()
#     results = chain.batch(samples)

        

import json
import os
from typing import Annotated, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# -------------------------- 核心配置（需根据实际情况调整） --------------------------
file_path = "processed_results.jsonl"          # 原始数据文件
output_path = "processed_results_with_subtext.jsonl"  # 最终输出文件
progress_path = "processing_progress.json"     # 进度记录文件（自动创建/更新）
batch_size = 9                                # 每批处理样本数
llm_api_key = os.getenv("DEEPSEEK_API_KEY")    # LLM API密钥

# -------------------------- 初始化依赖（LLM、解析器） --------------------------
# 初始化LLM
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key="sk-03be5006ffba43a49fe96f6727f797c9",
    max_tokens=4096,
    temperature=0.1,
    top_p=0.9
)

# 定义SubText数据结构
class SubText(BaseModel):
    noun: Annotated[str, "名词（单个单词）"]
    adjectives: Annotated[List[str], "形容词列表"]
    label: Annotated[int, "标签（1=正样本，0=负样本）"]
    sentence: Annotated[str, "自然短句（含名词和形容词）"]

# 初始化JSON解析器
parser = JsonOutputParser(pydantic_object=SubText)

# 构建LLM调用链
def get_chain():
    TEMPLATE = """
    {sample_str}
    请你找出此文本列表中所有的主体名词及其形容词以如下形式返回给我,不需要重复的，并有以下几条规则。
    1.如果某个名词没有形容词则不需要将该名词返回给我。
    2.其中第一条为正文本，由第一条构建的一组名词和形容词的label为1
    3.后面的所有文本为负文本，当有与第一条名词相同但是形容词不同时需返回，并且label为0
    4.负文本中与正文本相同的名词形容词组只需构建一次
    5.名词严格按照一个单词给出而不是名词组
    6.并将拆分出的名词和形容词构建为一个语法正确的短句,需要表达自然，必要时去掉and
    7.请严格按照我给出的规则返回json格式，不需要多余的内容
    {format_struction}
    示例：[{{"noun":"noun1","adjectives":["adj1"],"label":1,"sentence":"a photo containing a/an/the [adj1] noun."}},{{"noun":"noun2","adjectives":["adj1","adj2"],"label":0,"sentence":"a photo containing a/an/the [adj1],and [adj2] noun."}}]
    """
    prompt = PromptTemplate(
        template=TEMPLATE,
        input_variables=["sample_str"],
        partial_variables={"format_struction": parser.get_format_instructions()}
    )
    return prompt | llm | parser

# -------------------------- 进度管理工具函数 --------------------------
def load_processed_images():
    """加载已处理的image_name列表（从进度文件）"""
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)["processed_images"]
        except (json.JSONDecodeError, KeyError):
            print(f"进度文件 {progress_path} 格式异常，将重新创建")
    return []

def save_processed_images(processed_images):
    """保存已处理的image_name列表到进度文件"""
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({
            "processed_count": len(processed_images),
            "processed_images": processed_images,
            "last_update_time": os.popen("date").read().strip()  # 记录最后更新时间（Linux/Mac）
            # Windows系统替换为："last_update_time": os.popen("date /t && time /t").read().strip()
        }, f, ensure_ascii=False, indent=2)

# -------------------------- 主处理流程（支持断点续跑） --------------------------
def main():
    # 1. 加载已处理的样本（断点续跑核心）
    processed_images = load_processed_images()
    processed_count = len(processed_images)
    print(f"已处理样本数：{processed_count}（从 {progress_path} 加载）")

    # 2. 读取原始数据，筛选未处理的样本
    image_to_original = {}  # 原始数据映射：image_name → 原始JSON对象
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                image_to_original[obj["image_name"]] = obj
            except json.JSONDecodeError:
                print(f"跳过无效行：{line[:50]}...")
                continue

    # 筛选未处理的样本（排除已在processed_images中的）
    unprocessed_list = []
    for img_name, original_obj in image_to_original.items():
        if img_name not in processed_images:
            # 构建group列表（1个positive + 10个negatives的sentence）
            group = [original_obj["positive"]] + [neg["sentence"] for neg in original_obj["negatives"]]
            unprocessed_list.append({
                "image_name": img_name,
                "group": group,
                "original_obj": original_obj  # 直接携带原始对象，避免重复查找
            })

    total_unprocessed = len(unprocessed_list)
    print(f"待处理样本数：{total_unprocessed}，总样本数：{len(image_to_original)}")
    if total_unprocessed == 0:
        print("所有样本已处理完成，无需继续运行！")
        return

    # 3. 按批次处理未处理样本（实时保存+更新进度）
    for i in range(0, total_unprocessed, batch_size):
        batch = unprocessed_list[i:i+batch_size]
        batch_size_actual = len(batch)
        current_batch_num = i // batch_size + 1
        print(f"\n=== 处理第 {current_batch_num} 批（共 {batch_size_actual} 条）===")

        # 3.1 批量调用LLM生成subtext
        try:
            inputs = [{"sample_str": str(item["group"])} for item in batch]
            chain = get_chain()
            subtext_results = chain.batch(inputs)
            print(f"LLM调用成功，生成 {len(subtext_results)} 条subtext")
        except Exception as e:
            print(f"LLM调用失败：{str(e)[:100]}...，跳过该批")
            continue

        # 3.2 给原始对象添加subtext并收集结果
        batch_results = []
        newly_processed = []  # 记录当前批次已处理的image_name
        for idx in range(batch_size_actual):
            item = batch[idx]
            img_name = item["image_name"]
            subtext = subtext_results[idx]
            original_obj = item["original_obj"]

            # 复制原始对象（避免修改原映射）并添加subtext字段
            result_obj = original_obj.copy()
            result_obj["subtext"] = subtext
            batch_results.append(result_obj)
            newly_processed.append(img_name)  # 记录已处理的image_name

        # 3.3 追加保存当前批次结果到输出文件
        with open(output_path, "a", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"当前批保存成功，共 {len(batch_results)} 条写入 {output_path}")

        # 3.4 更新进度（关键：将新处理的image_name加入processed_images并保存）
        processed_images.extend(newly_processed)
        save_processed_images(processed_images)
        processed_count += len(newly_processed)
        print(f"进度更新成功：累计处理 {processed_count} 条，剩余 {total_unprocessed - (i + batch_size_actual)} 条")

    # 4. 处理完成提示
    print(f"\n=== 所有批次处理结束 ===")
    print(f"总处理样本数：{processed_count}")
    print(f"输出文件：{output_path}")
    print(f"进度文件：{progress_path}")

if __name__ == "__main__":
    main()