from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
import os
import langchain

# langchain.debug = True
class AttributeChange(BaseModel):
    attribute: str = Field(..., description="Attribute type")
    origin: str = Field(..., description="Original value")
    target: str = Field(..., description="Replacement value")

class NegativeExample(BaseModel):
    sentence: str = Field(..., description="Generated negative sample sentences")
    attribute_change: List[AttributeChange] = Field(..., description="List of attribute changes in generated negative sample sentences")

class DataItem(BaseModel):
    image_name: str = Field(..., description="Image name")
    positive: str = Field(..., description="Positive example sentence")
    negatives: List[NegativeExample] = Field(..., description="List of NegativeExample")

class RawData(BaseModel):
    image_name: str = Field(..., description="Image name")
    positive: str = Field(..., description="Positive example sentence")
    negatives: List[str] = Field(..., description="List of generated negative sample sentences")
     
RAW_PROMPT_TEMPLATE = """

Replace the attribute words in the given sentence(positive) and Generate 10 replaced sentences(negatives).
Replacement must strictly follow these rules:
1.Only replace the specified attribute words (Material/Pattern/Transparency/Color). Do not modify any other parts of the sentence.
2.The replacement attribute word must be of the same type as the original. For example, if replacing a Material type attribute word, only use a Material attribute word.
3.The replacement attribute word must be selected from the provided Attribute Options list.
4.Replace 1 to 3 attribute words per sentence. Do not exceed 3 replacements.
5.Replacement priority:
    1.Prioritize replacing Material, Pattern, and Transparency.
    2.After the above three types are replaced, then replace Color.
6.The replacement target attribute word must be different from the original attribute word. 

Attribute Options:
Material: plastic, metal, glass, wooden, fabric, leather, stone, ceramic, paper, wool, rattan, velvet, crochet
Pattern: logo, striped, woven, checkered, studded, floral, perforated, dotted, plain
Transparency: transparent, translucent, opaque
Color: black, white, grey, blue, green, red, brown, pink, purple, yellow, orange;

eg:
(The following situations represent the generation of errors:
    positive: Two fluffy white puppies with soft fur rest on vibrant green grass near a beige wall.
    negative: Two fluffy white puppies with soft fur rest on vibrant green grass near a wool wall.
    attribute_change=[AttributeChange(attribute='Material', origin='beige', target='wool')]
 Because beige does not exist in Attribute Options.Violation of rule 3.)
(The following situations represent the generation of errors:
    positive: Two white pelicans with long, orange beaks rest on green grass against a backdrop of dark foliage.
    negative: Two white pelicans with long, orange beaks rest on plastic grass against a backdrop of dark foliage.
    attribute_change=[AttributeChange(attribute='Material', origin='green', target='plastic')]
 Becauese green is not belong to Material attribute words.Violation of rule 2.)
(The following situations represent the generation of errors:
    positive: A leopard with a spotted coat carries a dark brown prey across a rocky surface in a natural setting.
    negative: A leopard with a spotted coat carries a dark brown prey across a rocky surface in a natural setting.
    attribute_change=[]
 Because the sentence has no changes.Violation of rule 6.)

{format_struction}

image name:
{image_name}
input sentence:
{input}

"""


PROMPT_TEMPLATE = """
我将提供你一个样本数据,其中包含正样本描述和负样本描述:
要求：
请你准确找出正样本到负样本中的属性的变换，将变化的属性分类(Material/Pattern/Transparency/Color)使用attribute记录，并将正确的词origin和变化后的词target记录。

属性词的类别请从下面给出的类别中去找，例如striped属于Pattern类:
Material: plastic, metal, glass, wooden, fabric, leather, stone, ceramic, paper, wool, rattan, velvet, crochet
Pattern: logo, striped, woven, checkered, studded, floral, perforated, dotted, plain
Transparency: transparent, translucent, opaque
Color: black, white, grey, blue, green, red, brown, pink, purple, yellow, orange;

{format_struction}

样本数据：
{input}
"""




def get_chain():
    #预生成负样本
    raw_llm = ChatTongyi(
        model="qwen-flash",
        model_kwargs={"temperature": 0.1,"enable_thinking": True  }
        )

    raw_parser =  PydanticOutputParser(pydantic_object=RawData)
    raw_prompt = PromptTemplate(
        input_variables=["input","image_name"],
        template=RAW_PROMPT_TEMPLATE,
        partial_variables={"format_struction": raw_parser.get_format_instructions()}
    )

    #处理生成的负样本
    llm = ChatTongyi(
        model="qwen-flash",
        model_kwargs={"temperature": 0.1,"enable_thinking": False  }
        )
    parser =  PydanticOutputParser(pydantic_object=DataItem)
    prompt = PromptTemplate(
        input_variables=["input"],
        template=PROMPT_TEMPLATE,
        partial_variables={"format_struction": parser.get_format_instructions()}
    )
    chain = raw_prompt | raw_llm | raw_parser  | prompt | llm | parser
    return chain


import json

def batch_read_jsonl(file_path, batch_size=10):
    """
    批次读取JSON Lines文件
    :param file_path: 文件路径
    :param batch_size: 每批读取的行数
    :return: 生成器，每次返回一个批次的数据（列表）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        batch = []  # 存储当前批次的数据
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)
                batch.append(data)
                # 当批次大小达到设定值时，返回该批次并清空
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError:
                print(f"跳过格式错误的行: {line}")
        # 处理最后一批（可能不足batch_size）
        if batch:
            yield batch


def get_processed_image_names(output_file):
    """读取已处理的结果文件，返回已处理过的 image_name 集合"""
    if not os.path.exists(output_file):
        return set()
    processed = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                processed.add(data.get("image_name"))
            except:
                continue
    return processed


if __name__ == "__main__":
    chain = get_chain()
    output_file = "processed_results.jsonl"
    processed_images = get_processed_image_names(output_file)
    print(f"✅ 已检测到 {len(processed_images)} 个已处理样本，将自动跳过。")

    # 自动记录断点文件
    checkpoint_file = "resume_checkpoint.txt"
    start_batch = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_batch = int(f.read().strip() or 0)
        print(f"🔄 从第 {start_batch+1} 批次继续处理。")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, batch in enumerate(batch_read_jsonl('./cleaned_image_descriptions.jsonl', batch_size=20)):
            if i < start_batch:
                continue  # 跳过已处理批次

            print(f"\n===== 第{i+1}批数据 =====")

            # 过滤掉已经处理过的样本
            batch = [item for item in batch if item["image_name"] not in processed_images]
            if not batch:
                print(f"第{i+1}批全部已处理，跳过。")
                continue

            inputs = [{"input": item["description"], "image_name": item["image_name"]} for item in batch]

            try:
                outputs = chain.batch(inputs)
                for result in outputs:
                    result_dict = result.model_dump()
                    f_out.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                    processed_images.add(result_dict["image_name"])
                f_out.flush()
                print(f"✅ 第{i+1}批处理完成。")

                # 写入断点
                with open(checkpoint_file, "w") as f_ckpt:
                    f_ckpt.write(str(i))

            except Exception as e:
                print(f"❌ 第{i+1}批处理出错：{str(e)}")
                print("程序将在下次启动时从该批次继续。")
                break

    print(f"🏁 所有批次处理完成，结果已保存至 {output_file}")
