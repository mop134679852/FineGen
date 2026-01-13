from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import base64
import json

load_dotenv()

# 配置本地vLLM端点
llm = ChatOpenAI(
    model_name="/data/mp/models/Qwen/Qwen2.5-VL-32B-Instruct",
    openai_api_base="http://0.0.0.0:8192/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    max_tokens=2048,
    temperature=0,
    top_p=0.9
)

# 将本地图像转换为base64编码
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 获取指定目录下的所有图像文件路径
def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.JPEG')
    image_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            image_paths.append(file_path)
    return image_paths

# 加载已处理的图像记录
def load_processed_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                return set()  # 文件损坏时重新开始
    return set()

# 保存已处理的图像记录
def save_processed_checkpoint(checkpoint_file, processed_images):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(processed_images), f, ensure_ascii=False)

# # 主逻辑
# if __name__ == "__main__":
#     # 配置参数
#     image_dir = r'/data/mp/datasets/ImageNet_nano'
#     output_file = "image_descriptions.jsonl"
#     checkpoint_file = "processed_checkpoint.json"  # checkpoint记录文件
#     batch_size = 10

#     # 加载已处理的图像
#     processed_images = load_processed_checkpoint(checkpoint_file)
#     print(f"已处理 {len(processed_images)} 张图像，将从剩余图像开始处理")

#     # 获取所有图像路径并过滤已处理的
#     all_image_paths = get_image_files(image_dir)
#     image_paths = []
#     image_names = []
#     for path in all_image_paths:
#         img_name = path.split('/')[-1]
#         if img_name not in processed_images:
#             image_paths.append(path)
#             image_names.append(img_name)
    
#     print(f"剩余待处理图像: {len(image_paths)} 张")
#     if not image_paths:
#         print("所有图像已处理完毕")
#         exit()

#     # 提示词模板
#     PROMPT_TEMPLATE = """Please describe the content in the image in English, using 10-25 words exactly.
# Must Include at least one attribute from color, material, pattern, or transparency, and all other relevant attributes from these categories. 

# Attribute options:
# Material: plastic, metal, glass, wood, fabric, leather, stone, ceramic, paper, text, wool, rattan, velvet, crochet
# Pattern: text, logo, striped, woven, checkered, studded, floral, perforated, dotted, plain
# Transparency: transparent, translucent, opaque
# Color: black, white, grey, blue, green, red, brown, pink, purple, yellow, orange; "light" and "dark" can only modify other colors (e.g., light blue, dark purple)

# Return only the English description—no extra text, explanations, or headings.
# """

#     # 构建消息列表
#     batch_messages = []
#     for path in image_paths:
#         image_base64 = image_to_base64(path)
#         message = [
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text": PROMPT_TEMPLATE},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
#                 ]
#             )
#         ]
#         batch_messages.append(message)

#     # 处理管道
#     chain = llm | StrOutputParser()
#     output = {}

#     # 批量处理并更新checkpoint
#     for i in range(0, len(batch_messages), batch_size):
#         batch = batch_messages[i:i+batch_size]
#         batch_names = image_names[i:i+batch_size]
        
#         try:
#             # 处理当前批次
#             responses = chain.batch(batch)
            
#             # 保存结果
#             for name, resp in zip(batch_names, responses):
#                 output[name] = resp
#                 processed_images.add(name)  # 标记为已处理
            
#             # 写入当前批次结果到输出文件（追加模式）
#             with open(output_file, "a", encoding="utf-8") as f:
#                 for name, desc in zip(batch_names, responses):
#                     json.dump({"image_name": name, "description": desc}, f, ensure_ascii=False)
#                     f.write("\n")
            
#             # 更新checkpoint
#             save_processed_checkpoint(checkpoint_file, processed_images)
#             print(f"已处理 {i+len(batch_names)}/{len(image_paths)} 张图像，已更新checkpoint")
        
#         except Exception as e:
#             print(f"处理批次时出错: {str(e)}")
#             print("当前进度已保存，可重新运行继续处理")
#             exit()

#     print("所有剩余图像处理完毕")


#     vllm serve /path/to/model \
# --dtype half \ # 半精度减少显存占用
# --tensor-parallel-size 2 \ # GPU数量
# --gpu-memory-utilization 0.8 \ # 显存利用率
# --max-model-len 2048 \ # 最大输入长度
# --max-num-seqs 8 \ # 最大并发序列数
# --enforce-eager # 禁用图优化，提升稳定性





# 配置本地vLLM端点
llm = ChatOpenAI(
    model_name="/data/mp/models/Qwen/Qwen2-VL-2B-LoRA-1-1000",
    openai_api_base="http://0.0.0.0:11113/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    max_tokens=2048,
    temperature=0.7,
    top_p=0.9
)

# 将本地图像转换为base64编码
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    


image_base64 = image_to_base64("/home/mp/llm_study/1.png")
message = [
    HumanMessage(
        content=[
            {"type": "text", "text": "Please convert the image to JSON"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    )
]



# 处理管道
chain = llm | StrOutputParser()

# 流式输出
for chunk in chain.stream(message):
    print(chunk, end="", flush=True)



