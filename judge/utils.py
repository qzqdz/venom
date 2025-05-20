import json
from pathlib import Path
from tqdm import tqdm
import openai
from config import (
    OPENAI_API_KEY, 
    OPENAI_MODEL, 
    OPENAI_TEMPERATURE, 
    MCQ_CONFIG,
    QUESTION_FIELD,
    CORRECT_ANSWER_FIELD,
    OPTIONS_FIELD
)

def load_qa_pairs(input_path):
    """Load QA pair data"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def h():
    """Get conversation history"""
    path = "history.json"
    record = ""
    try:
        with open(path, 'r', encoding='utf-8') as json_file:
            load = json.load(json_file)      
    except Exception as e:
        return record
    for i in load:
        record += i["role"] + ":" + i["content"] + "\n"
    return record 

def save_h(new_data, expert_name):
    """Save dialog to history"""
    fileroute = "history.json"
    try:
        with open(fileroute, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = []

    if not isinstance(existing_data, list):
        raise ValueError("Existing data in the file is not a list.")
    
    role = expert_name
    content = new_data
    n = {"role": role, "content": content}
    existing_data.append(n)
    with open(fileroute, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"Conversation successfully appended to {fileroute}")

def save_mcqs(output_path, mcqs):
    """Save generated multiple choice questions"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mcqs, f, ensure_ascii=False, indent=2)

def initialize_openai():
    """Initialize OpenAI client"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY,
                           base_url="https://api.siliconflow.cn/v1")
    return client

def generate_initial_question(client, question, answer):
    """Stage 1: Use OpenAI API to generate initial multiple choice question (question and correct answer only)"""
    prompt = MCQ_CONFIG["generation_prompt"].format(question=question, answer=answer)
    record = h()
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional question generation assistant who can transform simple QA pairs into high-quality multiple choice questions."},
                {"role": "user", "content": record+prompt}
            ],
            temperature=OPENAI_TEMPERATURE,
        )
        save_h(response.choices[0].message.content, "Question Generator")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error while generating initial question: {e}")
        return None

def parse_initial_question(response_text):
    """Parse initial question (question + correct answer)"""
    if not response_text:
        return None
    
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    result = {
        "question": "",
        "correct_option": "",
        "explanation": ""
    }
    
    for i, line in enumerate(lines):
        if line.startswith("问题:") or line.startswith("Question:"):
            result["question"] = line.replace("问题:", "").replace("Question:", "").strip()
        elif line.startswith("A:"):
            result["correct_option"] = line.replace("A:", "").strip()
        elif line.startswith("临时解析:") or line.startswith("Temporary Analysis:"):
            result["explanation"] = line.replace("临时解析:", "").replace("Temporary Analysis:", "").strip()
    
    # Process continuation lines for explanation
    for line in lines[lines.index(next(l for l in lines if "解析" in l or "Analysis" in l))+1:]:
        if not any(line.startswith(prefix) for prefix in ["问题:", "Question:", "A:", "B:", "C:", "D:"]):
            result["explanation"] += " " + line.strip()
    
    if not all([result["question"], result["correct_option"]]):
        print("Failed to parse initial question response")
        return None
    
    return result

def generate_distractors(client, initial_question):
    """Stage 2: Use OpenAI API to generate distractor options"""
    prompt = MCQ_CONFIG["distraction_prompt"].format(
        question=initial_question["question"],
        correct_option=initial_question["correct_option"],
        explanation=initial_question["explanation"]
    )
    record = h()
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional question generation assistant who can design high-quality distractor options for multiple choice questions."},
                {"role": "user", "content": record+prompt}
            ],
            temperature=OPENAI_TEMPERATURE,
        )
        save_h(response.choices[0].message.content, "Distractor Generator")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error while generating distractors: {e}")
        return None

def parse_distractors(response_text):
    """Parse distractor options response"""
    if not response_text:
        return None
    
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    result = {
        "options": {},
        "distractor_explanation": ""
    }
    
    current_option = None
    explanation_started = False
    explanation_lines = []
    
    for line in lines:
        if line.startswith("B:"):
            current_option = "B"
            result["options"]["B"] = line.replace("B:", "").strip()
        elif line.startswith("C:"):
            current_option = "C"
            result["options"]["C"] = line.replace("C:", "").strip()
        elif line.startswith("D:"):
            current_option = "D"
            result["options"]["D"] = line.replace("D:", "").strip()
        elif line.startswith("干扰项设计说明:") or line.startswith("Distractor Design Explanation:"):
            explanation_started = True
            explanation_lines.append(line.replace("干扰项设计说明:", "").replace("Distractor Design Explanation:", "").strip())
        elif explanation_started:
            explanation_lines.append(line)
    
    result["distractor_explanation"] = " ".join(explanation_lines)
    
    if len(result["options"]) != 3:
        print("Failed to parse all distractor options")
        return None
    
    return result

def combine_mcq_components(initial_question, distractors):
    """Combine initial question and distractors into a complete multiple choice question"""
    return {
        QUESTION_FIELD: initial_question["question"],
        OPTIONS_FIELD: {
            "A": initial_question["correct_option"],
            **distractors["options"]
        },
        CORRECT_ANSWER_FIELD: "A",
        "explanation": initial_question["explanation"],
        "distractor_explanation": distractors["distractor_explanation"]
    }

def review_mcq(client, mcq):
    """Stage 3: Review the quality of the multiple choice question"""
    prompt = MCQ_CONFIG["review_prompt"].format(
        question=mcq[QUESTION_FIELD],
        option_a=mcq[OPTIONS_FIELD]["A"],
        option_b=mcq[OPTIONS_FIELD]["B"],
        option_c=mcq[OPTIONS_FIELD]["C"],
        option_d=mcq[OPTIONS_FIELD]["D"],
        correct_answer=mcq[CORRECT_ANSWER_FIELD]
    )
    record = h()
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional question quality evaluator who can assess the effectiveness of multiple choice questions."},
                {"role": "user", "content": record+prompt}
            ],
            temperature=0.3,
        )
        review_text = response.choices[0].message.content
        save_h(review_text, "Question Evaluator")
        
        # Parse review results
        review_details = parse_review_response(review_text)
        is_approved = review_details.get("final_result", "").lower() == "pass" or review_details.get("total_score", 0) >= 18
        review_comment = review_details.get("assessment_opinion", "No detailed assessment provided.")
        
        return is_approved, review_comment, review_details
    except Exception as e:
        print(f"Error during question review: {e}")
        return False, str(e), {}

def parse_review_response(response_text):
    """Parse the review response text"""
    if not response_text:
        return {}
    
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    result = {}
    
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            
            if key in ["technical_precision", "knowledge_depth", "trap_discrimination", 
                      "lie_detection_effect", "authenticity"]:
                try:
                    result[key] = int(value.split()[0])
                except (ValueError, IndexError):
                    result[key] = 0
            elif key == "total_score":
                try:
                    result[key] = int(value.split()[0])
                except (ValueError, IndexError):
                    result[key] = sum(result.get(k, 0) for k in [
                        "technical_precision", "knowledge_depth", "trap_discrimination",
                        "lie_detection_effect", "authenticity"
                    ])
            else:
                result[key] = value
    
    return result

def clear():
    """Clear conversation history"""
    file_path = 'history.json'
    # Write empty array
    with open(file_path, 'w') as f:
        json.dump([], f)