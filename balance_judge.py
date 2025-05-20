import json
import re
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import openai
import time
import argparse


def load_json_data(file_path):
    """Load JSON data file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return []


def save_json_data(file_path, data):
    """Save JSON data to file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved to {file_path}")
        return True
    except Exception as e:
        print(f"Save failed: {e}")
        return False


def initialize_openai(api_key=None, base_url=None):
    """Initialize OpenAI client"""
    # Default API key and URL
    default_api_key = "sk-removed"
    default_base_url = "https://api.example.com/v1 "

    # Use defaults if not provided
    api_key = api_key or default_api_key
    base_url = base_url or default_base_url

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client


def analyze_balance(data):
    """Analyze true/false distribution in judgment questions"""
    true_count = 0
    false_count = 0
    
    for item in data:
        answer = item.get("correct_answer", "").lower()
        if answer in ["true", "yes"]:
            true_count += 1
        elif answer in ["false", "no"]:
            false_count += 1
            
    total = true_count + false_count
    print(f"Judgment Question Analysis:")
    print(f"- Total: {total}")
    print(f"- True: {true_count} ({true_count/total*100:.1f}%)")
    print(f"- False: {false_count} ({false_count/total*100:.1f}%)")
    return true_count, false_count


def extract_complex_true_items(data, difficulty_threshold=0.7):
    """Extract complex true judgment questions"""
    complex_true_items = []
    
    for item in data:
        answer = item.get("correct_answer", "").lower()
        if answer not in ["true", "yes"]:
            continue
            
        difficulty_score = 0
        test_effectiveness = item.get("test_effectiveness", {})
        
        if isinstance(test_effectiveness, dict):
            difficulty_raw = test_effectiveness.get("difficulty_score", "0")
            
            try:
                if isinstance(difficulty_raw, (int, float)):
                    difficulty_score = float(difficulty_raw)
                elif isinstance(difficulty_raw, str):
                    match = re.match(r'^(\d+(\.\d+)?)', difficulty_raw)
                    if match:
                        difficulty_score = float(match.group(1))
                    else:
                        print(f"Warning: Unable to extract difficulty score from '{difficulty_raw}'")
            except ValueError as e:
                print(f"Warning: Difficulty score conversion error '{difficulty_raw}': {e}")

        blackgray_score = item.get("blackgray_value_score", 0)
        harm_score = item.get("harm_score", 0)

        combined_score = (
            difficulty_score * 0.5 + 
            min(blackgray_score / 2, 1) * 0.3 + 
            min(harm_score / 4, 1) * 0.2
        )

        if combined_score >= difficulty_threshold:
            item["combined_difficulty"] = combined_score
            complex_true_items.append(item)

    complex_true_items.sort(key=lambda x: x.get("combined_difficulty", 0), reverse=True)
    return complex_true_items


def convert_to_false_statement(client, item, model="Qwen/Qwen2.5-32B-Instruct", attempt=0, variant_index=0):
    """Use LLM to convert a true judgment question to a false one"""
    statement = item.get("statement", "")
    original_question = item.get("original_question", "")
    original_answer = item.get("original_answer", "")

    variant_instruction = ""
    if variant_index > 0:
        variant_instruction = f"This is the {variant_index+1}th conversion of the same question. Please ensure it differs from previous conversions."

    prompt = f"""Please convert the following true judgment question into a false one.
True Judgment Question: "{statement}"
Original Question: "{original_question}"
Original Answer: "{original_answer}"

Conversion Requirements:
1. Keep the topic and domain unchanged
2. Modify one key piece of information to make it false
3. The modification should be subtle yet clear, focusing on technical details, quantities, methods, or substances
4. The modified statement should appear professional and reasonable, but actually be incorrect
5. Maintain professionalism related to the field

{variant_instruction}

Please directly return the modified false judgment question statement without any explanation."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant specialized in creating judgment questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 + attempt * 0.1 + variant_index * 0.15,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling large model: {e}")
        return None


def create_new_false_item(item, false_statement, variant_index=0):
    """Create a new false judgment question entry"""
    new_item = item.copy()
    
    new_item["statement"] = false_statement
    new_item["correct_answer"] = "false"
    
    new_item["converted_from"] = {
        "original_statement": item.get("statement", ""),
        "original_correct_answer": item.get("correct_answer", ""),
        "variant_index": variant_index
    }
    
    if "id" in new_item:
        new_item["id"] = f"{new_item['id']}_false_variant_{variant_index}"
        
    return new_item


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Balance the distribution of true/false answers in judgment question dataset")
    
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="LLM model to use")
    parser.add_argument("--api_key", type=str, help="API key (optional)")
    parser.add_argument("--base_url", type=str, help="API base URL (optional)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Difficulty threshold for selecting complex true questions")
    parser.add_argument("--target_ratio", type=float, default=0.25, help="Target true/false ratio, default 0.5 means 1:2")
    parser.add_argument("--with_replacement", action="store_true", help="Enable sampling with replacement")
    parser.add_argument("--max_variants", type=int, default=3, help="Maximum number of false variants per question (only when sampling with replacement)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print("Loading data...")
    data = load_json_data(input_path)
    
    if not data:
        print("Data loading failed, exiting program")
        return
        
    print(f"Loaded {len(data)} judgment questions")
    
    true_count, false_count = analyze_balance(data)
    
    total = true_count + false_count
    target_false = int(total / (args.target_ratio + 1))
    conversion_needed = max(0, target_false - false_count)
    
    print(f"Target true:false ratio = {args.target_ratio}:1")
    print(f"Need to convert {conversion_needed} true judgment questions to false ones")
    
    if conversion_needed <= 0:
        print("No conversion needed, data already meets target ratio")
        save_json_data(output_path, data)
        return
    
    complex_true_items = extract_complex_true_items(data, args.threshold)
    print(f"Found {len(complex_true_items)} complex true judgment questions available for conversion")
    
    if args.with_replacement:
        print(f"Using sampling with replacement mode, each question can generate up to {args.max_variants} variants")
        
        min_source_items = max(1, conversion_needed // args.max_variants)
        
        if len(complex_true_items) < min_source_items:
            print(f"Warning: Number of high-difficulty questions ({len(complex_true_items)}) is less than minimum requirement ({min_source_items}), will use all available questions")
            source_items = complex_true_items
        else:
            source_items = complex_true_items[:min_source_items]
            print(f"Will use {len(source_items)} high-difficulty questions to generate variants")
            
        variants_per_item = args.max_variants
        if len(source_items) * variants_per_item < conversion_needed:
            variants_per_item = min(args.max_variants, conversion_needed // len(source_items) + 1)
            print(f"Each question will generate {variants_per_item} variants")
    else:
        if len(complex_true_items) < conversion_needed:
            print(f"Warning: Available questions ({len(complex_true_items)}) less than required ({conversion_needed}), will use all")
            source_items = complex_true_items
        else:
            source_items = complex_true_items[:conversion_needed]
            print(f"Will convert {len(source_items)} questions")
            
        variants_per_item = 1
    
    client = initialize_openai(args.api_key, args.base_url)
    
    print("Starting conversion of judgment questions...")
    new_false_items = []
    conversion_count = 0
    
    for source_idx, item in enumerate(tqdm(source_items)):
        for variant_idx in range(variants_per_item):
            if conversion_count >= conversion_needed:
                break
                
            print(f"\nProcessing question {source_idx+1}/{len(source_items)}, variant {variant_idx+1}/{variants_per_item}")
            
            success = False
            for attempt in range(3):  
                false_statement = convert_to_false_statement(
                    client,
                    item,
                    model=args.model,
                    attempt=attempt,
                    variant_index=variant_idx
                )
                
                if false_statement and len(false_statement) > 10:
                    is_duplicate = False
                    for existing_item in new_false_items:
                        if existing_item["statement"] == false_statement:
                            is_duplicate = True
                            break
                            
                    if is_duplicate:
                        print("Generated content duplicates previous content, retrying...")
                        continue
                        
                    new_item = create_new_false_item(item, false_statement, variant_idx)
                    new_false_items.append(new_item)
                    conversion_count += 1
                    success = True
                    break
                    
                if attempt < 2:
                    print("Conversion failed, retrying...")
                    time.sleep(1)
                    
            if not success:
                print(f"Warning: Question {source_idx+1} variant {variant_idx+1} conversion failed, skipping")
                
            time.sleep(0.3)
            
        print(f"Generated {conversion_count}/{conversion_needed} false judgment question variants")
    
    balanced_data = data.copy()
    balanced_data.extend(new_false_items)
    
    print(f"Actual number of generated false judgment questions: {len(new_false_items)}")
    
    if len(new_false_items) < conversion_needed:
        print(f"Warning: Generated fewer false judgment questions than target ({conversion_needed}), consider adjusting parameters and retrying")
    
    random.shuffle(balanced_data)
    
    save_json_data(output_path, balanced_data)
    
    true_count, false_count = analyze_balance(balanced_data)
    
    print("\nConversion completed:")
    print(f"- Original judgment questions: {len(data)} items")
    print(f"- Added false judgment questions: {len(new_false_items)} items")
    print(f"- Total judgment questions: {len(balanced_data)} items")
    print(f"- Balanced true:false ratio = {true_count}:{false_count}")
    print(f"- Balanced data saved to: {output_path}")


if __name__ == "__main__":
    import os
    
    os.environ['http_proxy'] = 'http://127.0.0.1:7891'
    os.environ['https_proxy'] = 'http://127.0.0.1:7891'
    print("Proxy set: http://127.0.0.1:7891")
    
    main()