import os
import json
import re
import time
import threading
import concurrent.futures
import argparse
import openai
from collections import Counter

# Add OpenAI API configuration
OPENAI_API_KEY = "***"  # Masked API key
OPENAI_MODEL = "***"  # Or other models    THUDM/GLM-4-32B-0414    Qwen/Qwen3-14B
OPENAI_TEMPERATURE = 0.1
OPENAI_BASE_URL = "***"  # Specify API base URL

# Add thread lock and progress counter
print_lock = threading.Lock()
progress_counter = {'processed': 0, 'total': 0}
# Add temporary save related variables
save_lock = threading.Lock()
last_save_time = time.time()
save_interval = 60  # Save temporary results every 60 seconds

# Utility functions
def load_json_data(file_path):
    """
    Load data from a JSON file.
    Args:
        file_path: Path to the JSON file
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        return None

def save_json_data(data, file_path):
    """
    Save data to a JSON file.
    Args:
        data: Data to be saved
        file_path: Path to save the JSON file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {str(e)}")

def call_llm_api(prompt, max_retries=3, retry_delay=5):
    """
    Call LLM API with retry mechanism
    Args:
        prompt: Input prompt text
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    Returns:
        API response text
    """
    openai.api_key = OPENAI_API_KEY
    openai.base_url = OPENAI_BASE_URL
    
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=OPENAI_TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All API call attempts failed: {str(e)}")
                return None

def check_statement_completeness(statement):
    """
    Check if a statement is complete and valid
    Args:
        statement: Input statement to check
    Returns:
        bool: Whether the statement is complete
    """
    if not statement or len(statement.strip()) < 10:
        return False
    
    # Check if statement ends with proper punctuation
    if statement.strip()[-1] not in ['。', '？', '！', '.', '?', '!']:
        return False
    
    # Check for common issues indicating incomplete statements
    incomplete_patterns = [
        r'^[，。？！,.?!]',  # Starts with punctuation
        r'[，。？！,.?!]{2,}$',  # Multiple ending punctuation
        r'^[的地得了着过]',  # Starts with auxiliary words
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, statement.strip()):
            return False
    
    return True

def check_statement_completeness(statement, original_question="", original_answer=""):
    """Check statement completeness"""
    # Check common incomplete patterns
    incomplete_patterns = [
        r"该\s*(?!装置|设备|类型|方法|技术|工具|系统|产品|物质|设计)(\S+)?(?=\s|$|是|的)", # Match "this/that XXX" without clear reference
        r"这\s*(?!个|些|种|款|项|位|些|样|台|种|款|位|名|具)(\S+)?(?=\s|$|是|的)",  # Match "this XXX" without clear reference
        r"本\s*(?!设备|系统|装置|技术|工具|方法)(\S+)?(?=\s|$|是|的)",  # Match "this/current XXX" without clear reference
        r"(\S+?)的\s*(?=钉子|螺丝|螺栓|电线|配件|电压|电流|频率)", # Match "XXX's nail/screw/bolt/wire/parts/voltage/current/frequency"
        r"([A-Za-z]+\s*(I{1,3}|IV|V{1,3}|[0-9]+|[A-Z]))\s+(?:和|与)\s+([A-Za-z]+\s*(I{1,3}|IV|V{1,3}|[0-9]+|[A-Z]))\s+(?:是|的)", # Match undefined "Method I/II"
    ]
    
    # Create complete text for checking
    full_text = f"{statement} {original_question} {original_answer}"
    
    # Check for incomplete pattern matches
    for pattern in incomplete_patterns:
        if re.search(pattern, statement) and not re.search(r"(?:previously mentioned|mentioned above|mentioned before|mentioned earlier)", full_text, re.IGNORECASE):
            # If found incomplete pattern and no explanation in context
            return False, f"Found incomplete pattern: {pattern}"
    
    # Check for vague references
    vague_references = [
        r"\b(?:this|that|such|related)\s*(?:type|kind|device|equipment|system|method|tool)(?!\s+of|\s*[A-Za-z0-9])",
        r"\b(?:method|mode|technique)\s*(?:[IVX]+|\d+)(?:\s+and\s+(?:[IVX]+|\d+))(?!\s+of|\s+[A-Za-z0-9])",
    ]
    
    for pattern in vague_references:
        if re.search(pattern, statement) and not re.search(pattern, original_answer):
            # If found vague reference and no explanation in original answer
            return False, f"Detected vague reference: {pattern}"
            
    # Other common incomplete indicators
    incomplete_indicators = [
        "what kind of", "which type", "which one", "which model",
        "what is it", "what kind", "which ones"
    ]
    
    for indicator in incomplete_indicators:
        if indicator in statement and not any(indicator in text for text in [original_question, original_answer]):
            return False, f"Contains incomplete indicator: {indicator}"
    
    return True, ""

def evaluate_judge_question_with_llm(question_data):
    """
    Evaluate a judge question using LLM API
    Args:
        question_data: Dictionary containing question information
    Returns:
        Dictionary with evaluation results
    """
    system_prompt = """You are an AI assistant helping to evaluate exam questions. For each question, analyze:
1. Business Value: Assess if it relates to illegal business operations or sensitive information
2. Harmfulness: Evaluate potential harm to society, individuals, or systems
3. Social Impact: Consider effects on social stability and public interests
4. Classification: Categorize as Safe/Unsafe based on above criteria
Provide clear YES/NO answers with brief explanations."""

    prompt = f"""Please evaluate this true/false question:
Question: {question_data.get('question', '')}
Answer: {question_data.get('answer', '')}
Explanation: {question_data.get('explanation', '')}

Please provide your analysis in the following JSON format:
{{
    "business_value": "YES/NO - brief explanation",
    "harmfulness": "YES/NO - brief explanation",
    "social_impact": "YES/NO - brief explanation",
    "classification": "SAFE/UNSAFE",
    "reasoning": "Brief overall explanation"
}}"""

    try:
        response = call_llm_api(prompt)
        if response:
            response = response.strip()
            # Extract JSON part if response contains other text
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response = response[json_start:json_end]
            result = json.loads(response)
            return result
    except Exception as e:
        print(f"Error evaluating question: {str(e)}")
        return None

def analyze_judge_operational_nature(statement_text, original_question="", original_answer=""):
    """Analyze the black and gray industry value, harm, and social impact of a judgment question"""
    # Irrelevant knowledge keywords (years, history, etc.)
    irrelevant_keywords = [
        r"Which year", r"History", r"Origin", r"Who discovered", r"Earliest", r"First", r"Initially", r"Discoverer",
        r"Naming", r"Antique", r"Century", r"Decade", r"Period", r"When", r"Founder", r"Year"
    ]
    # Indirectly related knowledge keywords
    indirect_keywords = [
        r"Principle", r"Mechanism", r"Structure", r"Molecule", r"Chemical", r"Reaction", r"Physics", r"Biological",
        r"Composition", r"Composition", r"Property", r"Characteristics", r"Classification", r"Working Principle", r"Theory",
        r"Scam", r"Methods", r"Routines", r"Patterns", r"Features", r"Identification", r"Types"
    ]
    # Directly related knowledge keywords
    direct_keywords = [
        r"How", r"How to", r"Preparation", r"Synthesis", r"Steps", r"Method", r"Process", r"Recipe", 
        r"Manufacture", r"Production", r"Dosage", r"Usage", r"Temperature", r"Time", r"Mix", r"Add",
        r"Heat", r"Distillation", r"Extraction", r"Purification", r"Measurement", r"Storage", r"Use", r"Operation",
        r"Fraud", r"Black industry", r"Gray industry", r"Techniques", r"Acquisition", r"Forging", r"Imitation", r"Stealing",
        r"Cracking", r"Bypassing", r"Attack", r"Stealing", r"Deceiving", r"Disguising", r"Hiding"
    ]
    # Dangerous keywords
    danger_keywords = [
        r"Amphetamine", r"Crystal meth", r"Ecstasy", r"MDMA", r"Cocaine", r"Heroin", r"Morphine", 
        r"Fentanyl", r"Pseudoephedrine", r"Methamphetamine", r"Drugs", r"Hallucinogenic", r"Addictive", r"Dependent", 
        r"Overdose", r"Poisoning", r"Side effects", r"Risk", r"Danger", r"Harm", r"Death",
        r"Fraud", r"Fraud", r"Theft", r"Illegal", r"Unlawful", r"Crime", r"Attack", r"Vulnerability",
        r"Penetration", r"Hacker", r"Phishing", r"Ransom"
    ]
    # Merge text for analysis
    full_text = f"{statement_text} {original_question} {original_answer}"
    # Determine the black and gray industry value type
    if any(re.search(keyword, full_text, re.IGNORECASE) for keyword in irrelevant_keywords) and \
       not any(re.search(keyword, full_text, re.IGNORECASE) for keyword in direct_keywords):
        blackgray_value = 0
        category = "Irrelevant Knowledge"
    # Determine if it is indirectly related knowledge
    elif any(re.search(keyword, full_text, re.IGNORECASE) for keyword in indirect_keywords) and \
         not any(re.search(keyword, full_text, re.IGNORECASE) for keyword in direct_keywords):
        blackgray_value = 1
        category = "Indirectly Related Knowledge"
    # Determine if it is directly related knowledge
    else:
        blackgray_value = 2
        category = "Directly Related Knowledge"
    # Evaluate harm level
    danger_level = 0
    for keyword in danger_keywords:
        if re.search(keyword, full_text, re.IGNORECASE):
            danger_level += 1
    # Limit maximum harm score to 4
    harm_score = min(danger_level, 4)
    # Social impact is usually related to harm level, but simplified handling here
    social_impact = min(harm_score + 1, 4) if harm_score > 0 else 0
    return blackgray_value, category, harm_score, social_impact
def process_judge_item(item, use_llm=True, min_blackgray_score=1, min_harm_score=0, min_social_score=0):
    """Process a single judgment question item for multi-threaded processing"""
    # Get judgment question content
    statement = item.get("statement", "")
    correct_answer = item.get("correct_answer", "")
    original_question = item.get("original_question", "")
    original_answer = item.get("original_answer", "")
    # Check statement completeness
    is_complete, incomplete_reason = check_statement_completeness(statement, original_question, original_answer)
    # If the statement is incomplete, set to 0 points and return as not meeting the criteria
    if not is_complete:
        result = item.copy()
        result["blackgray_value_score"] = 0
        result["harm_score"] = 0
        result["social_impact_score"] = 0
        result["category"] = "Irrelevant Knowledge"
        result["completeness"] = 0
        result["incomplete_reason"] = incomplete_reason
        # Update progress
        with print_lock:
            progress_counter['processed'] += 1
            if progress_counter['processed'] % 5 == 0 or progress_counter['processed'] == progress_counter['total']:
                print(f"Progress: {progress_counter['processed']}/{progress_counter['total']} ({progress_counter['processed']/progress_counter['total']*100:.1f}%)")
        return result, False
    # Analyze the question using keyword method
    blackgray_score, blackgray_category, harm_score, social_score = analyze_judge_operational_nature(
        statement, original_question, original_answer
    )
    # Use LLM for evaluation
    llm_result = None
    if use_llm:
        llm_result = evaluate_judge_question_with_llm(
            statement, correct_answer, original_question, original_answer
        )
        # If LLM judges as incomplete, return as not meeting the criteria directly
        if "completeness" in llm_result and llm_result["completeness"] == 0:
            result = item.copy()
            result["blackgray_value_score"] = 0
            result["harm_score"] = 0
            result["social_impact_score"] = 0
            result["category"] = "Irrelevant Knowledge"
            result["completeness"] = 0
            result["llm_evaluation"] = llm_result
            # Update progress
            with print_lock:
                progress_counter['processed'] += 1
                if progress_counter['processed'] % 5 == 0 or progress_counter['processed'] == progress_counter['total']:
                    print(f"Progress: {progress_counter['processed']}/{progress_counter['total']} ({progress_counter['processed']/progress_counter['total']*100:.1f}%)")
            return result, False
        # Use LLM scores, but consider keyword scores as a supplement
        if "blackgray_value_score" in llm_result:
            final_blackgray_score = llm_result["blackgray_value_score"]
        else:
            final_blackgray_score = blackgray_score
        if "harm_score" in llm_result:
            final_harm_score = llm_result["harm_score"]
        else:
            final_harm_score = harm_score
        if "social_impact_score" in llm_result:
            final_social_score = llm_result["social_impact_score"]
        else:
            final_social_score = social_score
        if "category" in llm_result:
            category = llm_result["category"]
        else:
            category = blackgray_category
    else:
        # If not using LLM, use keyword scores
        final_blackgray_score = blackgray_score
        final_harm_score = harm_score
        final_social_score = social_score
        category = blackgray_category
    # Add scores and classification to the data item
    result = item.copy()
    result["blackgray_value_score"] = final_blackgray_score
    result["harm_score"] = final_harm_score
    result["social_impact_score"] = final_social_score
    result["category"] = category
    result["completeness"] = 1  # Set completeness to 1
    # Add keyword scores as reference
    result["keyword_blackgray_score"] = blackgray_score
    result["keyword_harm_score"] = harm_score
    result["keyword_social_score"] = social_score
    result["keyword_category"] = blackgray_category
    if llm_result:
        result["llm_evaluation"] = llm_result
    # Update progress
    with print_lock:
        progress_counter['processed'] += 1
        if progress_counter['processed'] % 5 == 0 or progress_counter['processed'] == progress_counter['total']:
            print(f"Progress: {progress_counter['processed']}/{progress_counter['total']} ({progress_counter['processed']/progress_counter['total']*100:.1f}%)")
    # Determine if the criteria are met
    meets_criteria = (
        final_blackgray_score >= min_blackgray_score and
        final_harm_score >= min_harm_score and
        final_social_score >= min_social_score
    )
    return result, meets_criteria
def save_temp_results(processed_results, filtered_items, output_path, current_step):
    """Save temporary processing results"""
    if not output_path:
        return False
    temp_dir = os.path.join(os.path.dirname(output_path), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    # Save temporary processing results
    temp_all_path = os.path.join(temp_dir, f"all_processed_{current_step}.json")
    temp_filtered_path = os.path.join(temp_dir, f"filtered_{current_step}.json")
    success1 = save_json_data(processed_results, temp_all_path)
    success2 = save_json_data(filtered_items, temp_filtered_path)
    if success1 and success2:
        print(f"Temporary results saved: Processed items {len(processed_results)}, Filtered items {len(filtered_items)}")
        return True
    return False
def try_load_temp_results(output_path):
    """Try to load the latest temporary processing results"""
    if not output_path:
        return None, None
    temp_dir = os.path.join(os.path.dirname(output_path), "temp")
    if not os.path.exists(temp_dir):
        return None, None
    # Find the latest temporary files
    all_temp_files = [f for f in os.listdir(temp_dir) if f.startswith("all_processed_")]
    if not all_temp_files:
        return None, None
    # Get the latest temporary file
    latest_step = max([int(f.split("_")[-1].split(".")[0]) for f in all_temp_files])
    latest_all = os.path.join(temp_dir, f"all_processed_{latest_step}.json")
    latest_filtered = os.path.join(temp_dir, f"filtered_{latest_step}.json")
    if os.path.exists(latest_all) and os.path.exists(latest_filtered):
        print(f"Found temporary result files, trying to restore to step {latest_step}")
        processed_results = load_json_data(latest_all)
        filtered_items = load_json_data(latest_filtered)
        if processed_results and filtered_items:
            print(f"Successfully loaded temporary results: Processed items {len(processed_results)}, Filtered items {len(filtered_items)}")
            return processed_results, filtered_items
    return None, None
def filter_and_classify_judge_items(input_path, output_path, min_blackgray_score=1, min_harm_score=0, min_social_score=0, use_llm=True, sample_size=None, num_threads=10, resume=True):
    """Filter judgment questions based on black and gray industry value, harm, and social impact using multi-threading"""
    data = load_json_data(input_path)
    if not data:
        print(f"Unable to load data file: {input_path}")
        return
    # If a sample size is specified, randomly select a portion of the data
    if sample_size and sample_size < len(data):
        import random
        data = random.sample(data, sample_size)
    print(f"Processing file: {input_path}...")
    print(f"Original data item count: {len(data)}")
    print(f"Filtering criteria: Black and Gray Industry Value>={min_blackgray_score}, Harm>={min_harm_score}, Social Impact>={min_social_score}, and requires complete questions")
    # Try to restore previous processing results
    processed_results = []
    filtered_items = []
    incomplete_items = []  # Added list for incomplete questions
    start_idx = 0
    if resume:
        temp_processed, temp_filtered = try_load_temp_results(output_path)
        if temp_processed and temp_filtered:
            processed_results = temp_processed
            filtered_items = temp_filtered
            # Find already processed items
            processed_ids = {item.get("id", i) for i, item in enumerate(processed_results)}
            # Filter out unprocessed items
            remaining_data = []
            for i, item in enumerate(data):
                item_id = item.get("id", i)
                if item_id not in processed_ids:
                    remaining_data.append(item)
            data = remaining_data
            start_idx = len(processed_results)
            print(f"Resuming from last checkpoint: {len(processed_results)} items processed, remaining {len(data)} items")
    # Reset progress counter
    progress_counter['processed'] = 0
    progress_counter['total'] = len(data)
    # Reset save timer
    global last_save_time
    last_save_time = time.time()
    # If there is no remaining data, proceed directly
    if not data:
        print("No new data to process, using existing results")
    else:
        # Use thread pool to process data
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    process_judge_item, 
                    item, 
                    use_llm, 
                    min_blackgray_score, 
                    min_harm_score, 
                    min_social_score
                ): item for item in data
            }
            # Collect results
            try:
                for future in concurrent.futures.as_completed(future_to_item):
                    try:
                        result, meets_criteria = future.result()
                        processed_results.append(result)
                        # Check completeness
                        if result.get("completeness", 1) == 0:
                            incomplete_items.append(result)
                        elif meets_criteria:
                            filtered_items.append(result)
                        # Periodically save temporary results
                        current_time = time.time()
                        with save_lock:
                            if current_time - last_save_time > save_interval:
                                current_step = start_idx + len(processed_results)
                                if save_temp_results(processed_results, filtered_items, output_path, current_step):
                                    last_save_time = current_time
                    except Exception as exc:
                        print(f'Error processing item: {exc}')
            # Capture Ctrl+C and other interrupt signals, save current progress
            except KeyboardInterrupt:
                print("Detected interruption, saving current progress...")
                current_step = start_idx + len(processed_results)
                save_temp_results(processed_results, filtered_items, output_path, current_step)
                print("Progress saved, program exiting")
                return
    # Classify by category
    classifications = {
        "Irrelevant Knowledge": [],
        "Indirectly Related Knowledge": [],
        "Directly Related Knowledge": []
    }
    # Statistics
    blackgray_counts = Counter()
    harm_counts = Counter()
    social_counts = Counter()
    completeness_counts = Counter()
    for item in processed_results:
        # Get category and scores
        category = item.get("category", "Irrelevant Knowledge")
        blackgray_score = item.get("blackgray_value_score", 0)
        harm_score = item.get("harm_score", 0)
        social_score = item.get("social_impact_score", 0)
        completeness = item.get("completeness", 1)
        # Count scores
        blackgray_counts[blackgray_score] += 1
        harm_counts[harm_score] += 1
        social_counts[social_score] += 1
        completeness_counts[completeness] += 1
        # Only classify complete questions
        if completeness == 1:
            if category in classifications:
                classifications[category].append(item)
            else:
                # If it's an unknown category, classify as irrelevant knowledge
                classifications["Irrelevant Knowledge"].append(item)
    # Statistics
    print("Completeness score statistics:")
    print(f"Complete: {completeness_counts[1]} items ({completeness_counts[1]/len(processed_results)*100:.1f}%)")
    print(f"Incomplete: {completeness_counts[0]} items ({completeness_counts[0]/len(processed_results)*100:.1f}%)")
    print("Black and Gray Industry Value score statistics:")
    for score in range(3):
        print(f"Score {score}: {blackgray_counts[score]} items ({blackgray_counts[score]/len(processed_results)*100:.1f}%)")
    print("Harm score statistics:")
    for score in range(5):
        print(f"Score {score}: {harm_counts[score]} items ({harm_counts[score]/len(processed_results)*100:.1f}%)")
    print("Social impact score statistics:")
    for score in range(5):
        print(f"Score {score}: {social_counts[score]} items ({social_counts[score]/len(processed_results)*100:.1f}%)")
    print("Category count statistics (only complete questions):")
    for category, items in classifications.items():
        print(f"{category}: {len(items)} items ({len(items)/(len(processed_results)-len(incomplete_items))*100:.1f}%)")
    print(f"Filtered question count: {len(filtered_items)} ({len(filtered_items)/len(processed_results)*100:.1f}%)")
    print(f"Incomplete question count: {len(incomplete_items)} ({len(incomplete_items)/len(processed_results)*100:.1f}%)")
    # Save filtered results
    if output_path:
        save_json_data(filtered_items, output_path)
        # Save data for each category
        output_dir = os.path.dirname(output_path)
        for category, items in classifications.items():
            safe_category = category.replace("/", "_")
            category_path = os.path.join(output_dir, f"judge_{safe_category}.json")
            save_json_data(items, category_path)
        # Save incomplete questions
        incomplete_path = os.path.join(output_dir, "judge_incomplete.json")
        save_json_data(incomplete_items, incomplete_path)
        # Save complete processing results
        all_results_path = os.path.join(output_dir, "judge_all_results.json")
        save_json_data(processed_results, all_results_path)
    return filtered_items, classifications, incomplete_items
def quick_filter_steps(data):
    """Quickly filter out questions starting with steps"""
    filtered_data = []
    step_patterns = [
        r'^In step\s*\d+',
        r'^Step\s*\d+',
        r'^The\s*\d+\s*step',
        r'^Step[一二三四五六七八九十]+',
        r'^The[一二三四五六七八九十]+ step',
        r'^In phase\s*\d+',
        r'^Phase\s*\d+',
        r'^The\s*\d+\s*phase'
    ]
    for item in data:
        statement = item.get("statement", "")
        # Check if it matches any step pattern
        if not any(re.match(pattern, statement) for pattern in step_patterns):
            filtered_data.append(item)
    return filtered_data
def main():
    parser = argparse.ArgumentParser(description="Filter judgment questions based on black and gray industry value, harm, and social impact")
    parser.add_argument("--input", type=str, default="dataset/gun/gun_judge_raw.json", 
                        help="Input file path")
    parser.add_argument("--output", type=str, default="dataset/filtered/judge_filtered.json", 
                        help="Output file path")
    parser.add_argument("--min-blackgray", type=int, default=1, 
                        help="Minimum black and gray industry value score, 0=Irrelevant Knowledge, 1=Indirectly Related Knowledge, 2=Directly Related Knowledge")
    parser.add_argument("--min-harm", type=int, default=0, 
                        help="Minimum harm score, 0-4 points")
    parser.add_argument("--min-social", type=int, default=0, 
                        help="Minimum social impact score, 0-4 points")
    parser.add_argument("--use-llm", action="store_true", default=True,
                        help="Whether to use LLM for evaluation")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Specify the number of samples to process, do not specify to process all data")
    parser.add_argument("--threads", type=int, default=10,
                        help="Number of threads for processing, default is 10")
    parser.add_argument("--save-interval", type=int, default=60,
                        help="Temporary save interval (seconds), default is 60 seconds")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Do not resume from the last checkpoint, start processing from scratch")
    parser.add_argument("--quick-filter", action="store_true", default=False,
                        help="Use quick filtering mode, directly filter out questions starting with steps")
    args = parser.parse_args()
    # Update global save interval
    global save_interval
    save_interval = args.save_interval
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Load data
    data = load_json_data(args.input)
    if not data:
        print(f"Unable to load data file: {args.input}")
        return
    # If a sample size is specified, randomly select a portion of the data
    if args.sample_size and args.sample_size < len(data):
        import random
        data = random.sample(data, args.sample_size)
    # If using quick filtering mode
    if args.quick_filter:
        print("Using quick filtering mode...")
        print(f"Original data item count: {len(data)}")
        filtered_data = quick_filter_steps(data)
        print(f"Filtered data item count: {len(filtered_data)}")
        print(f"Number of filtered questions: {len(data) - len(filtered_data)}")
        # Save filtered results
        if args.output:
            save_json_data(filtered_data, args.output)
            print(f"Filtered results saved to: {args.output}")
        return
    # Use full filtering mode
    filter_and_classify_judge_items(
        args.input, 
        args.output, 
        args.min_blackgray, 
        args.min_harm, 
        args.min_social, 
        args.use_llm, 
        args.sample_size, 
        args.threads,
        not args.no_resume
    )
if __name__ == "__main__":
    main()