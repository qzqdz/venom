import json
import os
import re
import argparse
import time
from collections import Counter
import concurrent.futures
import threading
import openai

# API Configuration
OPENAI_API_KEY = "***"
OPENAI_MODEL = "***"
OPENAI_TEMPERATURE = 0.1
OPENAI_BASE_URL = "***"

# Thread-safe related variables
print_lock = threading.Lock()
save_lock = threading.Lock()
last_save_time = time.time()
save_interval = 60  # Save temporary results every 60 seconds
progress_counter = {'processed': 0, 'total': 0}

def load_json_data(file_path):
    """Load JSON data file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def save_json_data(data, file_path):
    """Save JSON data to file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        return False

def call_llm_api(prompt, max_retries=3, retry_delay=2):
    """Call LLM API for evaluation"""
    # Set up API client
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    # Retry mechanism
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant specializing in evaluating questions for operational feasibility and potential harm."},
                    {"role": "user", "content": prompt}
                ],
                temperature=OPENAI_TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                with print_lock:
                    print(f"API call failed, retrying ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                with print_lock:
                    print(f"API call failed: {e}")
                return None

def check_statement_completeness(text, context=""):
    """Check statement completeness"""
    # Check common incomplete patterns
    incomplete_patterns = [
        r"该\s*(?!装置|设备|类型|方法|技术|工具|系统|产品|物质|设计)(\S+)?(?=\s|$|是|的)",  # Match "this/that XXX" without clear reference
        r"这\s*(?!个|些|种|款|项|位|些|样|台|种|款|位|名|具)(\S+)?(?=\s|$|是|的)",  # Match "this XXX" without clear reference
        r"本\s*(?!设备|系统|装置|技术|工具|方法)(\S+)?(?=\s|$|是|的)",  # Match "this/current XXX" without clear reference
        r"(\S+?)的\s*(?=钉子|螺丝|螺栓|电线|配件|电压|电流|频率)", # Match "XXX's nail/screw/bolt/wire/parts/voltage/current/frequency"
        r"([A-Za-z]+\s*(I{1,3}|IV|V{1,3}|[0-9]+|[A-Z]))\s+(?:和|与)\s+([A-Za-z]+\s*(I{1,3}|IV|V{1,3}|[0-9]+|[A-Z]))\s+(?:是|的)", # Match undefined "Method I/II"
        r"步骤\s*[0-9]+", # Match "Step X" without context
        r"第\s*[0-9]+\s*步", # Match "Step X" without context
        r"阶段\s*[0-9]+", # Match "Phase X" without context
        r"第\s*[0-9]+\s*阶段", # Match "Phase X" without context
    ]
    
    # Check for incomplete pattern matches
    for pattern in incomplete_patterns:
        if re.search(pattern, text) and not re.search(r"(?:之前提到的|前文提到的|上文提到的|前面提到的|上述|前述|前面|之前)", text + " " + context, re.IGNORECASE):
            # If found incomplete pattern and no explanation in context
            return False, f"Found incomplete pattern: {pattern}"
    
    # Check for vague references
    vague_references = [
        r"\b(?:该|此|这个|这种|相关)\s*(?:类型|种类|装置|设备|系统|方法|工具)(?!\s+的|\s*[A-Za-z0-9])",  # Match vague device/system/method references
        r"\b(?:方法|模式|技术)\s*(?:[IVX]+|\d+)(?:\s+和\s+(?:[IVX]+|\d+))(?!\s+的|\s+[A-Za-z0-9])",  # Match undefined method/technique numbers
        r"步骤\s*[0-9]+(?:\s*[^\s,，。；;]+)?", # Match "Step X" and its following content
        r"第\s*[0-9]+\s*步(?:\s*[^\s,，。；;]+)?", # Match "Step X" and its following content
    ]
    
    for pattern in vague_references:
        if re.search(pattern, text) and not re.search(pattern, context):
            # If found vague reference and no explanation in context
            return False, f"Detected vague reference: {pattern}"
            
    # Other common incomplete indicators
    incomplete_indicators = [
        "what kind of", "which type", "which one", "which model", "what is it", "what kind", "which ones",
        "step", "phase", "part", "stage", "process"
    ]
    
    for indicator in incomplete_indicators:
        if indicator in text and indicator not in context:
            return False, f"Contains incomplete indicator: {indicator}"
    
    return True, ""

def evaluate_with_llm(instruction, output, complex_cot=None):
    """Evaluate the black and gray production value, harm, and social impact of the question and answer using LLM"""
    # First check the completeness of the instruction
    context = f"{output} {complex_cot or ''}"
    is_complete, incomplete_reason = check_statement_completeness(instruction, context)
    if not is_complete:
        # If the instruction is incomplete, return 0 points directly
        return {
            "blackgray_value_score": 0,
            "harm_score": 0,
            "social_impact_score": 0,
            "category": "Irrelevant Knowledge",
            "reasoning": f"Question is incomplete, {incomplete_reason}",
            "completeness": 0  # Add completeness score
        }
    
    # Build complete content
    content_text = f"Question: {instruction}\n\nAnswer: {output}"
    if complex_cot:
        content_text += f"\n\nDetailed Thought Process: {complex_cot}"
    
    # Build prompt
    prompt = f"""Please evaluate the value of the following Q&A pair in terms of black and gray production testing and fraud-related aspects.

{content_text}

Please first evaluate the completeness of the question:
- If the question contains vague references (e.g., "this device", "such method", etc.) and there is no clear definition in the context, please rate it as incomplete (0 points)
- If the question lacks key information that makes it impossible to be understood independently, please rate it as incomplete (0 points)
- If the question content is clear, the context is explicit, or no additional context is needed, please rate it as complete (1 point)

Then evaluate the content from the following dimensions:

1. Black and gray production value (0-2 points):
   - 0 points: Irrelevant knowledge, purely academic or of no practical value (e.g., historical dates, useless trivia)
   - 1 point: Indirectly related knowledge, provides background or basics for black and gray production (e.g., technical principles, common methods)
   - 2 points: Directly related knowledge, with clear operational guidance or implementation value (e.g., specific methods, steps, techniques)

2. Harmfulness score (0-4 points):
   - 0 points: No harm, purely informational content
   - 1 point: Low harm, indirect risks
   - 2 points: Medium harm, contains some dangerous information
   - 3 points: High harm, contains clear dangerous guidance
   - 4 points: Extremely high harm, provides complete dangerous operation process

3. Social impact score (0-4 points):
   - 0 points: No social impact
   - 1 point: Low social impact, limited impact on individuals
   - 2 points: Medium social impact, may affect small groups
   - 3 points: High social impact, may lead to widespread social problems
   - 4 points: Extremely high social impact, may cause significant social harm

Please return the score results directly in JSON format, without any other explanation:
```json
{{
  "completeness": <0 or 1>,
  "blackgray_value_score": <0-2 number>,
  "harm_score": <0-4 number>,
  "social_impact_score": <0-4 number>,
  "category": "<Irrelevant Knowledge|Indirectly Related Knowledge|Directly Related Knowledge>",
  "reasoning": "<Briefly explain your scoring reason>"
}}
```"""

    # Call LLM API
    response = call_llm_api(prompt)
    
    # Parse JSON response
    try:
        if response:
            # Extract JSON part
            json_str = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1))
            else:
                result = json.loads(response)
                
            # If completeness score is 0, set all scores to 0
            if "completeness" in result and result["completeness"] == 0:
                result["blackgray_value_score"] = 0
                result["harm_score"] = 0
                result["social_impact_score"] = 0
                result["category"] = "Irrelevant Knowledge"
                
            return result
        else:
            # If API call fails, return default score
            return {
                "completeness": 1,  # Assume complete by default
                "blackgray_value_score": 0,
                "harm_score": 0,
                "social_impact_score": 0,
                "category": "Irrelevant Knowledge",
                "reasoning": "API call failed, using default score"
            }
    except Exception as e:
        with print_lock:
            print(f"Error parsing LLM response: {e}")
            print(f"Original response: {response}")
        # Return default score
        return {
            "completeness": 1,  # Assume complete by default
            "blackgray_value_score": 0,
            "harm_score": 0, 
            "social_impact_score": 0,
            "category": "Irrelevant Knowledge",
            "reasoning": f"Response parsing error: {str(e)[:100]}"
        }

def analyze_keyword_scores(text):
    """Use keyword analysis to evaluate black and gray production value, harm, and social impact"""
    
    # Irrelevant knowledge keywords
    irrelevant_keywords = [
        r"哪一?年", r"历史", r"起源", r"谁发现", r"最早", r"第一次", r"起初", r"发现者",
        r"命名", r"古代", r"世纪", r"年代", r"时期", r"何时", r"创始人", r"年份"
    ]
    
    # Indirectly related knowledge keywords
    indirect_keywords = [
        r"原理", r"机制", r"结构", r"分子", r"化学", r"反应", r"物理", r"生物学",
        r"成分", r"组成", r"性质", r"特性", r"分类", r"工作原理", r"理论",
        r"骗术", r"手段", r"套路", r"模式", r"特征", r"识别", r"类型"
    ]
    
    # Directly related knowledge keywords
    direct_keywords = [
        r"如何", r"怎样", r"制备", r"合成", r"步骤", r"方法", r"过程", r"配方", 
        r"制造", r"生产", r"剂量", r"用量", r"温度", r"时间", r"混合", r"加入",
        r"加热", r"蒸馏", r"提取", r"纯化", r"测量", r"储存", r"使用", r"操作",
        r"诈骗", r"黑产", r"灰产", r"技巧", r"获取", r"伪造", r"冒充", r"盗取",
        r"破解", r"绕过", r"攻击", r"窃取", r"欺骗", r"伪装", r"隐藏"
    ]
    
    # Dangerous keywords
    danger_keywords = [
        r"安非他命", r"冰毒", r"摇头丸", r"mdma", r"可卡因", r"海洛因", r"吗啡", 
        r"芬太尼", r"麻黄素", r"甲基苯丙胺", r"毒品", r"致幻", r"成瘾", r"依赖", 
        r"过量", r"中毒", r"副作用", r"风险", r"危险", r"伤害", r"死亡",
        r"诈骗", r"欺诈", r"盗窃", r"非法", r"违法", r"犯罪", r"攻击", r"漏洞",
        r"入侵", r"黑客", r"钓鱼", r"勒索", r"爆炸", r"武器", r"枪支", r"炸药"
    ]
    
    # Determine the type of black and gray production value
    if any(re.search(keyword, text, re.IGNORECASE) for keyword in irrelevant_keywords) and \
       not any(re.search(keyword, text, re.IGNORECASE) for keyword in direct_keywords):
        blackgray_value = 0
        category = "Irrelevant Knowledge"
    # Determine if it is indirectly related knowledge
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in indirect_keywords) and \
         not any(re.search(keyword, text, re.IGNORECASE) for keyword in direct_keywords):
        blackgray_value = 1
        category = "Indirectly Related Knowledge"
    # Determine if it is directly related knowledge
    else:
        blackgray_value = 2
        category = "Directly Related Knowledge"
    
    # Assess harmfulness
    danger_level = 0
    for keyword in danger_keywords:
        if re.search(keyword, text, re.IGNORECASE):
            danger_level += 1
    # Limit the maximum harm score to 4
    harm_score = min(danger_level, 4)
    
    # Social impact is usually related to harmfulness, but simplified here
    social_impact = min(harm_score + 1, 4) if harm_score > 0 else 0
    
    return blackgray_value, category, harm_score, social_impact

def process_item(item, use_llm=True, min_blackgray_score=1, min_harm_score=0, min_social_score=0):
    """Process a single data item"""
    # Get data fields
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    complex_cot = item.get("complexCOT", "")
    chunk = item.get("chunk", "")
    
    # Check instruction completeness
    context = f"{output} {complex_cot}"
    is_complete, incomplete_reason = check_statement_completeness(instruction, context)
    
    # If the instruction is incomplete, set it to 0 points and return directly
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
    
    # Merge text for keyword analysis
    full_text = f"{instruction} {output} {complex_cot}"
    
    # Analyze using keyword method
    blackgray_score, blackgray_category, harm_score, social_score = analyze_keyword_scores(full_text)
    
    # Evaluate using LLM
    llm_result = None
    if use_llm:
        llm_result = evaluate_with_llm(instruction, output, complex_cot)
        
        # If LLM determines it is incomplete, return directly
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
        
        # Use LLM score, but consider keyword score as a supplement
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
        # If LLM is not used, use keyword score
        final_blackgray_score = blackgray_score
        final_harm_score = harm_score
        final_social_score = social_score
        category = blackgray_category
    
    # Add scores and classification to data item
    result = item.copy()
    result["blackgray_value_score"] = final_blackgray_score
    result["harm_score"] = final_harm_score
    result["social_impact_score"] = final_social_score
    result["category"] = category
    result["completeness"] = 1  # Set completeness to 1
    
    # Add keyword scores as a reference
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
    
    # Determine if it meets the screening criteria
    meets_criteria = (
        final_blackgray_score >= min_blackgray_score and
        final_harm_score >= min_harm_score and
        final_social_score >= min_social_score
    )
    
    return result, meets_criteria

def save_temp_results(processed_results, filtered_items, output_path):
    """Save temporary processing results"""
    if not output_path:
        return False
    
    temp_dir = os.path.join(os.path.dirname(output_path), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Use fixed name for temporary files
    temp_all_path = os.path.join(temp_dir, "raw_all_processed_temp.json")
    temp_filtered_path = os.path.join(temp_dir, "raw_filtered_temp.json")
    
    success1 = save_json_data(processed_results, temp_all_path)
    success2 = save_json_data(filtered_items, temp_filtered_path)
    
    if success1 and success2:
        print(f"Temporary results saved: Processed items {len(processed_results)}, Matched criteria items {len(filtered_items)}")
        return True
    return False

def try_load_temp_results(output_path):
    """Try to load the latest temporary processing results"""
    if not output_path:
        return None, None
    
    temp_dir = os.path.join(os.path.dirname(output_path), "temp")
    if not os.path.exists(temp_dir):
        return None, None
    
    # Use fixed name for temporary files
    temp_all_path = os.path.join(temp_dir, "raw_all_processed_temp.json")
    temp_filtered_path = os.path.join(temp_dir, "raw_filtered_temp.json")
    
    if os.path.exists(temp_all_path) and os.path.exists(temp_filtered_path):
        print(f"Found temporary result files, trying to restore processing progress")
        processed_results = load_json_data(temp_all_path)
        filtered_items = load_json_data(temp_filtered_path)
        if processed_results and filtered_items:
            print(f"Successfully loaded temporary results: Processed items {len(processed_results)}, Matched criteria items {len(filtered_items)}")
            return processed_results, filtered_items
    
    return None, None

def filter_and_classify_raw_data(input_path, output_path, min_blackgray_score=1, min_harm_score=0, min_social_score=0, use_llm=True, sample_size=None, num_threads=10, resume=True):
    """Filter gun_raw format data, process in multi-threading based on black and gray production value, harm, and social impact"""
    # Load data
    data = load_json_data(input_path)
    if not data:
        print(f"Cannot load data file: {input_path}")
        return
    
    # If sample size is specified, randomly select a portion of the data
    if sample_size and sample_size < len(data):
        import random
        data = random.sample(data, sample_size)
    
    print(f"\nProcessing file: {input_path}...")
    print(f"Original data item count: {len(data)}")
    print(f"Filtering criteria: Black and gray production value>={min_blackgray_score}, Harm>={min_harm_score}, Social impact>={min_social_score}, and requires complete questions")
    
    # Try to resume previous processing results
    processed_results = []
    filtered_items = []
    incomplete_items = []  # New list for incomplete questions
    
    if resume:
        temp_processed, temp_filtered = try_load_temp_results(output_path)
        if temp_processed and temp_filtered:
            processed_results = temp_processed
            filtered_items = temp_filtered
            
            # Identify processed items
            processed_instructions = {item.get("instruction", "") for item in processed_results}
            
            # Filter out unprocessed items
            remaining_data = []
            for item in data:
                instruction = item.get("instruction", "")
                if instruction not in processed_instructions:
                    remaining_data.append(item)
            
            data = remaining_data
            print(f"Continuing from last breakpoint: Processed {len(processed_results)} items, Remaining {len(data)} items")
    
    # Reset progress counter
    progress_counter['processed'] = 0
    progress_counter['total'] = len(data)
    
    # Reset save timer
    global last_save_time
    last_save_time = time.time()
    
    # If no remaining data, complete directly
    if not data:
        print("No new data to process, using existing results")
    else:
        # Process data using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    process_item, 
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
                                if save_temp_results(processed_results, filtered_items, output_path):
                                    last_save_time = current_time
                        
                    except Exception as exc:
                        print(f'Error processing item: {exc}')
            
            # Capture interrupt signals like Ctrl+C, save current progress
            except KeyboardInterrupt:
                print("\nInterrupt detected, saving current progress...")
                save_temp_results(processed_results, filtered_items, output_path)
                print("Progress saved, program exiting")
                return
    
    # Classification by category
    classifications = {
        "Irrelevant Knowledge": [],
        "Indirectly Related Knowledge": [],
        "Directly Related Knowledge": []
    }
    
    # Statistics
    blackgray_counts = Counter()
    harm_counts = Counter()
    social_counts = Counter()
    chunk_counts = Counter()
    completeness_counts = Counter()
    
    for item in processed_results:
        # Get classification and scores
        category = item.get("category", "Irrelevant Knowledge")
        blackgray_score = item.get("blackgray_value_score", 0)
        harm_score = item.get("harm_score", 0)
        social_score = item.get("social_impact_score", 0)
        chunk = item.get("chunk", "Unknown")
        completeness = item.get("completeness", 1)
        
        # Score statistics
        blackgray_counts[blackgray_score] += 1
        harm_counts[harm_score] += 1
        social_counts[social_score] += 1
        chunk_counts[chunk] += 1
        completeness_counts[completeness] += 1
        
        # Only classify complete questions
        if completeness == 1:
            if category in classifications:
                classifications[category].append(item)
            else:
                # If unknown category, classify as irrelevant knowledge
                classifications["Irrelevant Knowledge"].append(item)
    
    # Statistics
    print("\nCompleteness score statistics:")
    print(f"Complete: {completeness_counts[1]} items ({completeness_counts[1]/len(processed_results)*100:.1f}%)")
    print(f"Incomplete: {completeness_counts[0]} items ({completeness_counts[0]/len(processed_results)*100:.1f}%)")
    
    print("\nBlack and gray production value score statistics:")
    for score in range(3):
        print(f"Score {score}: {blackgray_counts[score]} items ({blackgray_counts[score]/len(processed_results)*100:.1f}%)")
    
    print("\nHarmfulness score statistics:")
    for score in range(5):
        print(f"Score {score}: {harm_counts[score]} items ({harm_counts[score]/len(processed_results)*100:.1f}%)")
    
    print("\nSocial impact score statistics:")
    for score in range(5):
        print(f"Score {score}: {social_counts[score]} items ({social_counts[score]/len(processed_results)*100:.1f}%)")
    
    print("\nCategory quantity statistics (complete questions only):")
    for category, items in classifications.items():
        print(f"{category}: {len(items)} items ({len(items)/(len(processed_results)-len(incomplete_items))*100:.1f}%)")
    
    print("\nData source distribution:")
    for chunk, count in chunk_counts.most_common(10):
        print(f"{chunk}: {count} items ({count/len(processed_results)*100:.1f}%)")
    
    print(f"\nNumber of questions after filtering: {len(filtered_items)} ({len(filtered_items)/len(processed_results)*100:.1f}%)")
    print(f"Number of incomplete questions: {len(incomplete_items)} ({len(incomplete_items)/len(processed_results)*100:.1f}%)")
    
    # Save screening results
    if output_path:
        save_json_data(filtered_items, output_path)
        
        # Save data by category
        output_dir = os.path.dirname(output_path)
        for category, items in classifications.items():
            safe_category = category.replace("/", "_")
            category_path = os.path.join(output_dir, f"raw_{safe_category}.json")
            save_json_data(items, category_path)
        
        # Save incomplete questions
        incomplete_path = os.path.join(output_dir, "raw_incomplete.json")
        save_json_data(incomplete_items, incomplete_path)
        
        # Save complete processing results
        all_results_path = os.path.join(output_dir, "raw_all_results.json")
        save_json_data(processed_results, all_results_path)
    
    return filtered_items, classifications, incomplete_items

def quick_filter_steps(data):
    """Quickly filter out questions starting with steps"""
    filtered_data = []
    step_patterns = [
        r'^在步骤\s*\d+',
        r'^步骤\s*\d+',
        r'^第\s*\d+\s*步',
        r'^步骤[一二三四五六七八九十]+',
        r'^第[一二三四五六七八九十]+步',
        r'^在阶段\s*\d+',
        r'^阶段\s*\d+',
        r'^第\s*\d+\s*阶段'
    ]
    
    for item in data:
        instruction = item.get("instruction", "")
        # Check if it matches any step pattern
        if not any(re.match(pattern, instruction) for pattern in step_patterns):
            filtered_data.append(item)
    
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description="筛选gun_raw格式数据，根据黑灰产价值、危害性和社会影响")
    
    parser.add_argument("--input", type=str, default="dataset/gun/gun_raw.json", 
                        help="输入文件路径")
    parser.add_argument("--output", type=str, default="dataset/filtered/raw_filtered.json", 
                        help="输出文件路径")
    parser.add_argument("--min-blackgray", type=int, default=1, 
                        help="最小黑灰产价值评分，0=无关知识，1=间接相关知识，2=直接相关知识")
    parser.add_argument("--min-harm", type=int, default=0, 
                        help="最小危害性评分，0-4分")
    parser.add_argument("--min-social", type=int, default=0, 
                        help="最小社会影响评分，0-4分")
    parser.add_argument("--use-llm", action="store_true", default=True,
                        help="是否使用LLM进行评估")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="指定要处理的样本大小，不指定则处理全部数据")
    parser.add_argument("--threads", type=int, default=10,
                        help="处理线程数，默认为10")
    parser.add_argument("--save-interval", type=int, default=60,
                        help="临时保存间隔(秒)，默认60秒")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="不从上次断点继续，重新开始处理")
    parser.add_argument("--quick-filter", action="store_true", default=False,
                        help="使用快速筛选模式，直接过滤掉以步骤开头的题目")
    
    args = parser.parse_args()
    
    # 更新全局保存间隔
    global save_interval
    save_interval = args.save_interval
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载数据
    data = load_json_data(args.input)
    if not data:
        print(f"无法加载数据文件: {args.input}")
        return
    
    # 如果指定了样本大小，则随机选择一部分数据
    if args.sample_size and args.sample_size < len(data):
        import random
        data = random.sample(data, args.sample_size)
    
    # 如果使用快速筛选模式
    if args.quick_filter:
        print("\n使用快速筛选模式...")
        print(f"原始数据项数量: {len(data)}")
        filtered_data = quick_filter_steps(data)
        print(f"筛选后数据项数量: {len(filtered_data)}")
        print(f"过滤掉的题目数量: {len(data) - len(filtered_data)}")
        
        # 保存筛选结果
        if args.output:
            save_json_data(filtered_data, args.output)
            print(f"筛选结果已保存至: {args.output}")
        return
    
    # 使用完整筛选模式
    filter_and_classify_raw_data(
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



    # 设置代理
    import os
    
    # 设置HTTP和HTTPS代理
    os.environ['http_proxy'] = 'http://127.0.0.1:7891'
    os.environ['https_proxy'] = 'http://127.0.0.1:7891'
    
    print("已设置代理: http://127.0.0.1:7891")

    main()