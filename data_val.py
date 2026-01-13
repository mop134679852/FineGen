import json
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import logging
import time
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
import langchain

langchain.debug = True

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class ValidationState(TypedDict):
    image_name: Annotated[str, "图片名称"]
    positive: Annotated[str, "正样本描述"]
    negative: Annotated[Dict[str,Any],"当前处理的负样本"]
    attribute_changes: Annotated[List[Dict[str,str]],"属性变更列表"]
    validation_result: Annotated[bool,"验证结果"]
    correction_count: Annotated[int,"纠正次数"]
    error_reason: Annotated[str,"错误原因"]
    corrected_data: Annotated[Optional[Dict[str,Any]],"纠正后的数据"]
    current_step: Annotated[str,"当前步骤"]
    validation_details: Annotated[str,"验证详细说明"]

class LLMValidationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="/data/mp/models/Qwen/Qwen3-32B",
            openai_api_base="http://0.0.0.0:10086/v1",
            openai_api_key="None",
            max_tokens=2048,
            temperature=0,
            top_p=0.9,
            extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
    }
            )
    def validate(self,state:ValidationState) -> Dict[str,Any]:

        
        positive = state["positive"]
        if state["correction_count"] <= 0:
            negative_sentence = state["negative"]["sentence"]
            attribute_changes = state["attribute_changes"]
        else:
            negative_sentence = state["corrected_data"]["sentence"]
            attribute_changes = state["corrected_data"]["attribute_change"]
        class vaildatation_info(BaseModel):
            is_valid: bool = Field(..., description="验证合理性")
            error_reason: str = Field(..., description="错误原因,明确说明原因（例如：属性变更没有产生有意义的语义差异。）")
            details: str = Field(..., description="详细分析")
        parser = PydanticOutputParser(pydantic_object=vaildatation_info)
        # 构建验证提示
        system_prompt = """你是一个严格的数据验证专家。请根据以下规则验证属性变更的合理性：

        验证规则：
        1. 替换的属性词必须与原始词类型相同。例如，如果替换材质类型属性词，只能使用材质属性词。
        2. 替换的目标属性词必须与原始属性词不同。
        3. 原始属性词必须在正样本描述中实际存在。
        4. 属性变更应该产生有意义的语义差异。

        请严格分析，如果违反任何规则，请明确指出具体违反的规则和原因。"""

        user_prompt = f"""
        请验证以下数据：

        正样本: {positive}
        负样本: {negative_sentence}
        属性变更: {attribute_changes}

        请按以下格式回答：
        {parser.get_format_instructions()}
        

        如果属性变更列表为空，请直接判定为不合理。"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        chain = self.llm | parser

        # response = self.llm.invoke(messages)
        response = chain.invoke(messages)

        if response.is_valid == True:
            state["validation_result"] = True
            state["error_reason"] = ""
            state["validation_details"] = response.details
            return state
        else:
            state["validation_result"] = False
            state["error_reason"] = response.error_reason
            state["validation_details"] = response.details
            return state

# 基于LLM的纠正Agent
class LLMCorrectionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="/data/mp/models/Qwen/Qwen3-32B",
            openai_api_base="http://0.0.0.0:10086/v1",
            openai_api_key="None",
            max_tokens=2048,
            temperature=0,
            top_p=0.9,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
    }
            )
    
    def correct(self, state: ValidationState) -> Dict[str, Any]:
        """使用LLM进行智能纠正"""

        class AttributeChange(BaseModel):
            attribute: str = Field(..., description="纠正后的属性类型")
            origin: str = Field(..., description="纠正后的源属性词")
            target: str = Field(..., description="纠正后的目标属性词")

        class NegativeExample(BaseModel):
            sentence: str = Field(..., description="纠正后的负样本")
            attribute_change: List[AttributeChange] = Field(..., description="纠正后的属性变化列表")
        
        parser = PydanticOutputParser(pydantic_object=NegativeExample)

        positive = state["positive"]
        current_negative = state["negative"]["sentence"]
        attribute_changes = state["attribute_changes"]
        error_reason = state["error_reason"]
        validation_details = state["validation_details"]
        
        system_prompt = """你是一个数据纠正专家。请根据验证错误和属性选项，智能地纠正属性变更。

                    纠正原则：
                    1. 保持属性类型一致
                    2. 确保产生有意义的语义变化
                    3. 保持句子的语法和逻辑正确性
                    4. 纠正原本的错误
                    5. 正样本的属性必须和原来一致
                    """

        user_prompt = f"""
        请纠正以下不合理的属性变更：

        正样本: {positive}
        当前负样本: {current_negative}
        原属性变更: {attribute_changes}
        错误原因: {error_reason}
        验证细节：{validation_details}

        请提供纠正后的属性变更列表，并生成新的负样本句子。

        请按以下格式回答：
        {parser.get_format_instructions()}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        chain = self.llm | parser
        response = chain.invoke(messages)
        
        # 解析纠正结果
        corrected_changes = [c.model_dump() for c in response.attribute_change]  # 转成 JSON 可序列化的 dict
        new_negative = response.sentence

        state["corrected_data"] = {
            "sentence": new_negative,
            "attribute_change": corrected_changes
        }
        state["attribute_changes"] = corrected_changes
        state["correction_count"] += 1
        return state
# 创建图节点函数
def validate_node(state: ValidationState) -> Dict[str, Any]:
    """验证节点"""
    logger.info(f"开始验证: {state['image_name']}")
    
    validator = LLMValidationAgent()
    result = validator.validate(state)
    
    logger.info(f"验证结果: {'通过' if result['validation_result'] else '不通过'} - {result.get('error_reason', '')}")
    state["current_step"] = "validated"
    state["error_reason"] = result.get("error_reason", "")
    state["validation_result"] = result["validation_result"]
    state["validation_details"] = result.get("validation_details", "")
    return state
def correct_node(state: ValidationState) -> Dict[str, Any]:
    """纠正节点"""
    logger.info(f"开始第{state['correction_count'] + 1}次纠正")
    
    corrector = LLMCorrectionAgent()
    result = corrector.correct(state)
    
    logger.info(f"第{result['correction_count']}次纠正完成")
    state["current_step"] = "corrected"
    state["correction_count"] = result["correction_count"]
    state["corrected_data"] = result["corrected_data"]
    return state

def route_after_validation(state: ValidationState) -> str:
    """验证后的路由逻辑"""
    if state["validation_result"]:
        return "accept"
    elif state["correction_count"] < 3:  # 最多纠正3次
        return "correct"
    else:
        return "reject"



def accept_node(state: ValidationState) -> Dict[str, Any]:
    """接受数据节点"""
    logger.info(f"数据验证通过: {state['image_name']}")
    return {"current_step": "accepted"}

def reject_node(state: ValidationState) -> Dict[str, Any]:
    """拒绝数据节点"""
    logger.info(f"数据被拒绝（超过最大纠正次数）: {state['image_name']}")
    return {"current_step": "rejected"}

# 构建状态图
def create_validation_graph():
    """创建验证工作流图"""
    builder = StateGraph(ValidationState)
    
    # 添加节点
    builder.add_node("validate", validate_node)
    builder.add_node("correct", correct_node)
    builder.add_node("accept", accept_node)
    builder.add_node("reject", reject_node)
    
    # 设置入口点
    builder.set_entry_point("validate")
    
    # 添加条件边
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "accept": "accept",
            "correct": "correct", 
            "reject": "reject"
        }
    )
    
    # 添加从纠正回到验证的边
    builder.add_edge("correct", "validate")
    
    # 添加终止边
    builder.add_edge("accept", END)
    builder.add_edge("reject", END)
    
    return builder.compile()

# 主处理类
class DataValidator:
    def __init__(self):
        self.graph = create_validation_graph()
        self.results = {
            "accepted": [],
            "corrected": [],
            "rejected": []
        }
        self.statistics = {
            "total_processed": 0,
            "total_accepted": 0,
            "total_corrected": 0,
            "total_rejected": 0
        }
    
    def process_jsonl_file(self, input_file: str, output_file: str, batch_size: int = 8):
        """以 batch_size 为单位批量处理样本，充分利用 GPU 并行计算"""
        logger.info(f"开始处理文件: {input_file}，batch_size={batch_size}")

        buffer = []
        with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

            for i, line in enumerate(f_in):
                data = json.loads(line.strip())
                buffer.append(data)

                # 当缓存满一个 batch，就一起跑
                if len(buffer) >= batch_size:
                    results = self.process_batch(buffer)
                    for r in results:
                        f_out.write(json.dumps(r, ensure_ascii=False) + '\n')
                    f_out.flush()
                    buffer.clear()
            
            # 处理最后一批不足 batch_size 的数据
            if buffer:
                results = self.process_batch(buffer)
                for r in results:
                    f_out.write(json.dumps(r, ensure_ascii=False) + '\n')
                f_out.flush()

        self._print_summary()
        return self.statistics

    def process_batch(self, data_batch: List[dict]) -> List[dict]:
        """一次性处理一个批次（多张图片、多条样本）"""
        all_initial_states = []

        # 组装所有样本的验证任务
        for data in data_batch:
            image_name = data["image_name"]
            positive = data["positive"]
            for negative in data["negatives"]:
                all_initial_states.append({
                    "image_name": image_name,
                    "positive": positive,
                    "negative": negative,
                    "attribute_changes": negative.get("attribute_change", []),
                    "validation_result": False,
                    "correction_count": 0,
                    "error_reason": "",
                    "corrected_data": {},
                    "current_step": "start",
                    "validation_details": ""
                })

        logger.info(f"批量开始验证，共 {len(all_initial_states)} 条负样本")

        # 一次性送入 GPU（LangGraph 内部会并行执行 validate/correct）
        final_states = self.graph.batch(all_initial_states)
        logger.info("批量验证完成")

        # 汇总结果（与 process_single_data 类似）
        grouped_results = {}
        for fs in final_states:
            img = fs["image_name"]
            if img not in grouped_results:
                grouped_results[img] = {
                    "image_name": img,
                    "positive": fs["positive"],
                    "negatives": [],
                    "processing_stats": {
                        "total_negatives": 0,
                        "accepted_negatives": 0,
                        "corrected_negatives": 0,
                        "rejected_negatives": 0
                    }
                }

            group = grouped_results[img]
            group["processing_stats"]["total_negatives"] += 1
            neg = fs["negative"]

            if fs.get("corrected_data"):
                corrected_negative = neg.copy()
                corrected_negative.update(fs["corrected_data"])
                group["negatives"].append(corrected_negative)
                group["processing_stats"]["corrected_negatives"] += 1
            elif fs["current_step"] == "accepted":
                group["negatives"].append(neg)
                group["processing_stats"]["accepted_negatives"] += 1
            elif fs["current_step"] == "rejected":
                group["processing_stats"]["rejected_negatives"] += 1

        final_clean_results = []
        for img_data in grouped_results.values():
            clean_entry = {
                "image_name": img_data["image_name"],
                "positive": img_data["positive"],
                "negatives": img_data["negatives"]
            }
            final_clean_results.append(clean_entry)

        return final_clean_results
    
    def _print_summary(self):
        """打印处理摘要"""
        total = self.statistics["total_processed"]
        accepted = self.statistics["total_accepted"]
        corrected = self.statistics["total_corrected"]
        rejected = self.statistics["total_rejected"]
        
        logger.info("=" * 60)
        logger.info("数据处理摘要:")
        logger.info(f"总处理负样本数: {total}")
        logger.info(f"直接接受数: {accepted} ({accepted/total*100:.1f}%)")
        logger.info(f"纠正后接受数: {corrected} ({corrected/total*100:.1f}%)")
        logger.info(f"拒绝数: {rejected} ({rejected/total*100:.1f}%)") 
        logger.info(f"总接受率: {(accepted + corrected)/total*100:.1f}%")
        logger.info("=" * 60)
# 使用示例
if __name__ == "__main__":
    # 创建验证器实例
    validator = DataValidator()
    
    # 处理文件
    input_file = "processed_results.jsonl"
    output_file = "validated_results.jsonl"
    
    try:
        results = validator.process_jsonl_file(input_file, output_file,batch_size=1)
        logger.info("数据处理完成！")
        # logger.info(f"处理统计: {json.dumps(results, indent=2)}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")