# Import necessary modules
import json
import os
from pathlib import Path
from tqdm import tqdm
from utils import (
    load_qa_pairs,
    save_mcqs,
    initialize_openai,
    generate_initial_question,
    parse_initial_question,
    generate_distractors,
    parse_distractors,
    combine_mcq_components,
    review_mcq,
    clear
)
from config import (
    QUESTION_FIELD,
    CORRECT_ANSWER_FIELD,
    ORIGINAL_QUESTION_FIELD,
    ORIGINAL_ANSWER_FIELD,
    OPTIONS_FIELD
)

def clear_history():
    """Clear conversation history"""
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
    """Save results to file in real-time"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved to {file_path}")
    except Exception as e:
        print(f"Save failed: {e}")

def process_question(client, question, answer, processed_questions, accepted_mcqs, rejected_mcqs, output_path, rejected_path):
    """Process three-stage workflow for a single question"""
    # Skip already processed questions
    if question in processed_questions:
        print(f"Skipping processed question: {question[:30]}...")
        return

    # Check if question and answer are valid
    if not question or not answer:
        print(f"Skipping invalid QA pair: Question={question[:30]}... Answer={answer[:30]}...")
        return
    
    # Stage 1: Generate initial question (question and correct answer)
    print(f"\nStep 1/3: Generating initial question (testing knowledge internalization)...")
    initial_response = generate_initial_question(client, question, answer)
    initial_question = parse_initial_question(initial_response)
    
    if not initial_question:
        print("Initial question generation failed, skipping this question")
        add_to_rejected(question, answer, "Initial question generation failed", rejected_mcqs, rejected_path)
        return
    
    # Stage 2: Generate distractors
    print(f"Step 2/3: Generating professional distractors...")
    distractors_response = generate_distractors(client, initial_question)
    distractors = parse_distractors(distractors_response)
    
    if not distractors:
        print("Distractor generation failed, skipping this question")
        add_to_rejected(question, answer, "Distractor generation failed", rejected_mcqs, rejected_path)
        return
    
    # Combine into complete multiple choice question
    mcq = combine_mcq_components(initial_question, distractors)
    
    # Stage 3: Quality assessment
    print(f"Step 3/3: Evaluating question discrimination...")
    is_approved, review_comment, review_details = review_mcq(client, mcq)
    
    # Handle based on evaluation results
    if is_approved:
        print(f"Question passed quality assessment! Total score: {review_details.get('total_score', 'Unknown')}")
        # Build complete multiple choice question object
        formatted_mcq = {
            QUESTION_FIELD: mcq[QUESTION_FIELD],
            OPTIONS_FIELD: mcq[OPTIONS_FIELD],
            CORRECT_ANSWER_FIELD: mcq[CORRECT_ANSWER_FIELD],
            ORIGINAL_QUESTION_FIELD: question,
            ORIGINAL_ANSWER_FIELD: answer,
            "explanation": mcq.get("explanation", ""),
            "distractor_explanation": mcq.get("distractor_explanation", ""),
            "review_comment": review_comment,
            "review_scores": review_details.get("scores", {}),
            "total_score": review_details.get("total_score", 0)
        }
        accepted_mcqs.append(formatted_mcq)
        save_results_realtime(output_path, accepted_mcqs)
    else:
        print(f"Question did not pass evaluation. Reason: {review_comment}")
        add_to_rejected(
            question, 
            answer, 
            f"Did not pass evaluation: {review_comment}", 
            rejected_mcqs, 
            rejected_path, 
            partial_mcq=mcq,
            review_details=review_details
        )

def add_to_rejected(question, answer, reason, rejected_mcqs, rejected_path, partial_mcq=None, review_details=None):
    """Add to rejected list and save"""
    rejected_item = {
        ORIGINAL_QUESTION_FIELD: question,
        ORIGINAL_ANSWER_FIELD: answer,
        "reason": reason
    }
    
    # Add partially generated multiple choice question information if available
    if partial_mcq:
        rejected_item["partial_mcq"] = partial_mcq
    
    # Add review details if available
    if review_details:
        rejected_item["review_details"] = review_details
    
    rejected_mcqs.append(rejected_item)
    save_results_realtime(rejected_path, rejected_mcqs)

def main():
    # Initialize
    client = initialize_openai()
    
    # Input and output paths
    # input_path = Path("data/origin.json")
    # output_path = Path("data/output_mcqs.json")
    # rejected_path = Path("data/rejected_mcqs.json")

    # input_path = Path("../data/drug/merged_data.json")   
    # output_path = Path("data/harmful_eval_results.json")
    # rejected_path = Path("data/harmful_rejected_results.json")

    # input_path = Path("../data/gun.json")   
    # output_path = Path("data/harmful_eval_results.json")
    # rejected_path = Path("data/harmful_rejected_results.json")

    input_path = Path("../data/drug_new_f.json")   
    output_path = Path("data/drug_new_choice.json")
    rejected_path = Path("data/drug_new_choice_rejected.json")


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
    print("Loading QA pair data...")
    qa_pairs = load_qa_pairs(input_path)
    print(f"Loaded {len(qa_pairs)} QA pairs, processed {len(processed_questions)} pairs")
    
    # Generate multiple choice questions
    print("Starting to generate and evaluate knowledge multiple choice questions using three-stage processing...")
    
    for idx, qa in enumerate(tqdm(qa_pairs, desc="Processing")):
        question = qa.get("instruction", "")
        answer = qa.get("output", "")
        
        print(f"\nProcessing question {idx+1}/{len(qa_pairs)}: {question[:50]}...")
        
        # Three-stage question processing
        process_question(
            client, 
            question, 
            answer,
            processed_questions,
            accepted_mcqs,
            rejected_mcqs,
            output_path,
            rejected_path
        )
        
        # Mark question as processed
        processed_questions.add(question)
        
        # Clear conversation history to avoid context confusion
        clear()
    
    # Final result saving
    save_mcqs(output_path, accepted_mcqs)
    save_mcqs(rejected_path, rejected_mcqs)
    
    # Calculate statistics
    total_processed = len(accepted_mcqs) + len(rejected_mcqs)
    acceptance_rate = len(accepted_mcqs) / total_processed * 100 if total_processed > 0 else 0
    
    # Calculate average score
    if accepted_mcqs:
        avg_score = sum(item.get("total_score", 0) for item in accepted_mcqs) / len(accepted_mcqs)
    else:
        avg_score = 0
    
    # Statistics by scoring dimension
    dimension_scores = {}
    if accepted_mcqs:
        for mcq in accepted_mcqs:
            for dimension, score in mcq.get("review_scores", {}).items():
                if dimension not in dimension_scores:
                    dimension_scores[dimension] = []
                dimension_scores[dimension].append(score)
    
    avg_dimension_scores = {
        dim: sum(scores) / len(scores) for dim, scores in dimension_scores.items() if scores
    }
    
    print(f"\nEvaluation complete! Statistics:")
    print(f"- Total QA pairs processed: {len(qa_pairs)}")
    print(f"- High-quality multiple choice questions generated: {len(accepted_mcqs)} ({acceptance_rate:.1f}%)")
    print(f"- Rejected questions: {len(rejected_mcqs)}")
    print(f"- Average score of accepted questions: {avg_score:.2f}/25")
    print("\nAverage score by dimension:")
    
    # New scoring dimension names
    dimension_mapping = {
        "技术精确性": "Technical Accuracy",
        "知识深度": "Knowledge Depth",
        "陷阱区分度": "Distractor Distinction",
        "测谎效果": "Lie Detection Effect",
        "真实性": "Authenticity",
        # Compatible with old fields
        "信息层次得分": "Technical Accuracy",
        "独立性得分": "Knowledge Depth",
        "教学价值得分": "Lie Detection Effect",
        "可复用性得分": "Authenticity",
        "选项质量得分": "Distractor Distinction"
    }
    
    # Display organized by new scoring dimensions
    expected_dimensions = ["Technical Accuracy", "Knowledge Depth", "Distractor Distinction", "Lie Detection Effect", "Authenticity"]
    for dim in expected_dimensions:
        scores = []
        for old_dim, old_scores in dimension_scores.items():
            if dimension_mapping.get(old_dim) == dim:
                scores.extend(old_scores)
        
        if scores:
            avg = sum(scores) / len(scores)
            print(f"- {dim}: {avg:.2f}/5")
        else:
            print(f"- {dim}: No data")
    
    print(f"\nResults saved to:")
    print(f"- Accepted multiple choice questions: {output_path}")
    print(f"- Rejected multiple choice questions: {rejected_path}")

if __name__ == "__main__":


    import os
    import sys
    
    # Set proxy
    os.environ['http_proxy'] = 'http://127.0.0.1:7891'
    os.environ['https_proxy'] = 'http://127.0.0.1:7891'
    
    print("Proxy set to: http://127.0.0.1:7891")


    main()