import json
import os
from pathlib import Path
from tqdm import tqdm
from utils import (
    load_qa_pairs,
    save_mcqs,
    initialize_openai,
    generate_mcq,
    parse_mcq_response,
    review_mcq,
    morecheck,
    analyze_confusability,
    parse_confusability_response,
)

# Define field name constants
STATEMENT_FIELD = "statement"
CORRECT_ANSWER_FIELD = "correct_answer"
ORIGINAL_QUESTION_FIELD = "original_question"
ORIGINAL_ANSWER_FIELD = "original_answer"

def clear():
    file_path = 'history.json'
    # Write empty array
    with open(file_path, 'w') as f:
        json.dump([], f)

def load_existing_results(file_path):
    """Load existing result data"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read existing results: {e}, creating new file")
    return []

def save_results_realtime(file_path, data_list):
    """Save results to specified file in real time"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"Saved in real time to {file_path}")
    except Exception as e:
        print(f"Save failed: {e}")

def main():
    # Initialize
    client = initialize_openai()
    
    # Input output paths
    # input_path = Path("../data/drug/merged_data.json")   
    # output_path = Path("data/harmful_eval_results.json")
    # rejected_path = Path("data/harmful_rejected_results.json")

    # 输入输出路径
    input_path = Path("../data/drug_new_f.json")   
    output_path = Path("data/drug_new_judge.json")
    rejected_path = Path("data/drug_new_judge_rejected.json")
    # recovery_path = Path("data/drug_new_judge_recovery.json")

    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True)
    
    # Load existing results
    accepted_mcqs = load_existing_results(output_path)
    rejected_mcqs = load_existing_results(rejected_path)
    
    # Record processed questions to avoid duplicates
    processed_questions = {
        item.get(ORIGINAL_QUESTION_FIELD, "") for item in accepted_mcqs
    }.union({
        item.get(ORIGINAL_QUESTION_FIELD, "") for item in rejected_mcqs
    })
    
    # Load QA pairs
    print("Loading harmful knowledge QA pair data...")
    qa_pairs = load_qa_pairs(input_path)
    print(f"Loaded {len(qa_pairs)} questions, processed {len(processed_questions)}")
    
    # Generate true/false questions
    print("Starting to generate and review true/false questions for evaluating LLM harmful capabilities...")
    
    for qa in tqdm(qa_pairs, desc="Processing"):
        question = qa.get("instruction", "")
        answer = qa.get("output", "")
        
        # Skip processed questions
        if question in processed_questions:
            print(f"Skipping processed question: {question[:30]}...")
            continue
        
        if not question or not answer:
            print(f"Skipping invalid QA pair: {qa}")
            continue
            
        # Try to generate and review, max 3 retries
        for attempt in range(3):
            # Generate true/false question
            response_text = generate_mcq(client, question, answer)
            mcq = parse_mcq_response(response_text)
            
            if not mcq:
                continue
            
            # 混淆性分析
            confusability_result_text = analyze_confusability(client, mcq[STATEMENT_FIELD], mcq[CORRECT_ANSWER_FIELD])
            print(f"\n混淆性分析原始结果: {confusability_result_text}")
            
            # 解析混淆性分析结果
            confusability_result = parse_confusability_response(confusability_result_text)
            print(f"测评难度: {confusability_result.get('测评难度', 'N/A')}")
            
            # 审核判断题质量
            is_approved, review_comment = review_mcq(client, mcq, question, answer)
            print(f"审核结果: {'通过' if is_approved else '不通过'}, 评语: {review_comment}")
            
            # 如果审核通过，保存结果并跳出循环
            if is_approved:
                # 按照指定格式保存
                formatted_mcq = {
                    STATEMENT_FIELD: mcq[STATEMENT_FIELD],
                    CORRECT_ANSWER_FIELD: mcq[CORRECT_ANSWER_FIELD],
                    ORIGINAL_QUESTION_FIELD: question,
                    ORIGINAL_ANSWER_FIELD: answer,
                    "test_effectiveness": {
                        "context_completeness": confusability_result.get("上下文完整性", "不完整"),
                        "relevance": confusability_result.get("相关性", "不相关"),
                        "clarity": confusability_result.get("表述明确性", "不明确"),
                        "difficulty_score": confusability_result.get("测评难度", "0.5"),
                        "review_comment": review_comment
                    }
                }
                accepted_mcqs.append(formatted_mcq)
                # 实时保存通过的题目
                save_results_realtime(output_path, accepted_mcqs)
                break
        
        # 如果所有尝试都失败，添加到拒绝列表
        else:
            rejected_item = {
                ORIGINAL_QUESTION_FIELD: question,
                ORIGINAL_ANSWER_FIELD: answer,
                "reason": "无法生成符合有害内容测评要求的题目"
            }
            rejected_mcqs.append(rejected_item)
            # 实时保存被拒绝的题目
            save_results_realtime(rejected_path, rejected_mcqs)
        
        # 将问题标记为已处理
        processed_questions.add(question)
        
        clear()
    
    print(f"\n已完成! 共处理 {len(qa_pairs)} 个有害知识问答对")
    print(f"- 生成有效测评判断题: {len(accepted_mcqs)} 个")
    print(f"- 被拒绝的低质量判断题: {len(rejected_mcqs)} 个")
    print(f"- 已保存到 {output_path} 和 {rejected_path}")

if __name__ == "__main__":
    main()