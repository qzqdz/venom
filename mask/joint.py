# Import required modules
from openai import OpenAI
import json
import random
from typing import Dict, Any
from tqdm import tqdm

def replace_with_probability(text: str, mask_mapping: Dict[str, Dict[str, str]], keep_harmful_prob: float, tolerance: float = 0.05) -> tuple[str, float]:
    """
    Replace sensitive content in text based on probability while controlling the actual proportion of harmful content
    :param text: Original text
    :param mask_mapping: Sensitive word mapping
    :param keep_harmful_prob: Probability of keeping harmful content (0-1)
    :param tolerance: Allowed error range (default ±5%)
    :return: (Replaced text, Actual harmful ratio)
    """
    result = text
    sensitive_to_mask = mask_mapping["sensitive_to_mask"]
    mask_to_harmless = mask_mapping["mask_to_harmless"]
    
    # Sort sensitive terms by length to ensure longer terms are replaced first
    sorted_terms = sorted(sensitive_to_mask.keys(), key=len, reverse=True)
    
    # Count occurrences of each sensitive term
    term_counts = {term: text.count(term) for term in sorted_terms}
    total_term_count = sum(term_counts.values())
    
    if total_term_count == 0:
        return result, 0
        
    # Calculate number of sensitive terms to replace
    target_replace_count = int(total_term_count * (1 - keep_harmful_prob))
    
    # Randomly shuffle sensitive terms while maintaining length priority
    length_groups = {}
    for term in sorted_terms:
        term_len = len(term)
        if term_len not in length_groups:
            length_groups[term_len] = []
        length_groups[term_len].append(term)
    
    # Shuffle terms within each length group
    for terms in length_groups.values():
        random.shuffle(terms)
    
    # Track number of replaced sensitive terms
    replaced_count = 0
    
    # Process sensitive terms by length
    for length in sorted(length_groups.keys(), reverse=True):
        for term in length_groups[length]:
            # Skip if term doesn't appear
            if term_counts[term] == 0:
                continue
                
            # Calculate ratio after replacing this term
            potential_replaced = replaced_count + term_counts[term]
            potential_ratio = 1 - (potential_replaced / total_term_count)
            
            # Skip if replacement would cause ratio to deviate too much from target
            if abs(potential_ratio - keep_harmful_prob) > tolerance:
                continue
                
            # Replace the sensitive term
            mask_code = sensitive_to_mask[term]
            harmless_text = mask_to_harmless[mask_code]
            result = result.replace(term, harmless_text)
            replaced_count = potential_replaced
            
            # Stop replacing if target ratio is reached within tolerance
            current_ratio = 1 - (replaced_count / total_term_count)
            if abs(current_ratio - keep_harmful_prob) <= tolerance:
                return result, current_ratio
    
    # Return final result and actual ratio
    final_ratio = 1 - (replaced_count / total_term_count)
    return result, final_ratio

def generate_malicious_qa(input_data: Dict[str, Any], api_key: str, keep_harmful_prob: float = 0.5) -> Dict[str, Any]:
    """
    Generate QA pairs with controlled proportion of harmful content
    :param input_data: Input data containing original QA and mask mapping
    :param api_key: API key (no longer used)
    :param keep_harmful_prob: Probability of keeping harmful content (0-1)
    :return: Generated results
    """
    # Get original text and mapping
    original_text = input_data['original_output']
    masked_text = input_data['masked_output']
    sensitive_to_mask = input_data['mask_mapping']['sensitive_to_mask']
    mask_to_harmless = input_data['mask_mapping']['mask_to_harmless']
    
    # Get prefix and suffix
    prefix = input_data.get('prefix', '')  # Use empty string if no prefix
    suffix = input_data.get('suffix', '')  # Use empty string if no suffix
    
    # Count occurrences of each sensitive term
    term_counts = {term: original_text.count(term) for term in sensitive_to_mask.keys()}
    total_terms = sum(term_counts.values())
    
    if total_terms == 0:
        return {
            "original_instruction": input_data["original_instruction"],
            "original_output": original_text,
            "malicious_output": f"{prefix}\n\n{masked_text}\n\n{suffix}",  # If no sensitive words, use harmless version directly
            "target_harmful_ratio": keep_harmful_prob,
            "actual_harmful_ratio": 0,
            "total_sensitive_terms": 0,
            "remaining_sensitive_terms": 0,
            "harm_score": input_data.get("harm_score", 0),
            "social_impact_score": input_data.get("social_impact_score", 0)
        }
    
    # Sort sensitive terms by length to ensure longer terms are replaced first
    sorted_terms = sorted(sensitive_to_mask.keys(), key=len, reverse=True)
    
    # Calculate number of sensitive terms to replace
    target_replace_count = int(total_terms * (1 - keep_harmful_prob))
    
    # Randomly select sensitive terms to replace
    terms_to_replace = []
    current_count = 0
    
    # Randomly shuffle sensitive terms
    random.shuffle(sorted_terms)
    
    # Select sensitive terms to replace
    for term in sorted_terms:
        if current_count >= target_replace_count:
            break
        count = term_counts[term]
        if count > 0:
            terms_to_replace.append(term)
            current_count += count
    
    # Perform replacement
    result_text = original_text
    for term in terms_to_replace:
        mask = sensitive_to_mask[term]
        harmless = mask_to_harmless[mask]
        result_text = result_text.replace(term, harmless)
    
    # Count remaining sensitive terms
    remaining_terms = sum(result_text.count(term) for term in sensitive_to_mask.keys())
    actual_ratio = remaining_terms / total_terms if total_terms > 0 else 0
    
    # Build final output (add prefix and suffix)
    final_output = f"{prefix}\n\n{result_text}\n\n{suffix}"
    
    # Return result
    return {
        "original_instruction": input_data["original_instruction"],
        "original_output": original_text,
        "malicious_output": final_output,
        "target_harmful_ratio": keep_harmful_prob,
        "actual_harmful_ratio": actual_ratio,
        "total_sensitive_terms": total_terms,
        "remaining_sensitive_terms": remaining_terms,
        "harm_score": input_data.get("harm_score", 0),
        "social_impact_score": input_data.get("social_impact_score", 0)
    }

def fix_jsonl_format(content: str) -> str:
    """
    Fix JSONL file format
    :param content: Original file content
    :return: Fixed content
    """
    # Split into lines
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Remove trailing comma
        if line.endswith(','):
            line = line[:-1]
            
        # Ensure each line is valid JSON
        try:
            json.loads(line)
            fixed_lines.append(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON line detected: {line[:50]}... ({str(e)})")
            continue
            
    return '\n'.join(fixed_lines)

def process_json_file(input_file: str, output_file: str, api_key: str, keep_harmful_prob: float = 0.5, sample_count: int = 1, tolerance: float = 0.05):
    """
    Process JSON file containing multiple QA pairs, supports sampling with replacement for specified number of samples
    Supports both standard JSON and JSONL formats
    :param input_file: Input JSON file path
    :param output_file: Output JSON file path
    :param api_key: API key
    :param keep_harmful_prob: Probability of keeping harmful content (0-1)
    :param sample_count: Number of samples to generate
    :param tolerance: Allowed error range (default ±5%)
    """
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check file format
        print("Checking file format...")
        if content.strip().startswith('[') and content.strip().endswith(']'):
            # JSON array format
            try:
                data = json.loads(content)
                print("Detected JSON array format")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON array format: {str(e)}")
                return
        else:
            # Attempt to fix JSONL format
            print("Attempting to fix JSONL format...")
            fixed_content = fix_jsonl_format(content)
            
            # Write fixed content to temporary file
            fixed_file = input_file + '.fixed'
            with open(fixed_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed file saved to: {fixed_file}")
            
            # Read fixed content
            data = []
            for line in fixed_content.split('\n'):
                if line.strip():
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid line: {line[:50]}... ({str(e)})")
                        continue
    
    except Exception as e:
        print(f"Failed to read file: {str(e)}")
        return

    if not data:
        print("No valid data found")
        return

    print(f"Successfully read {len(data)} records")
    
    # Data validation
    valid_data = []
    for item in data:
        if isinstance(item, dict) and 'original_instruction' in item and 'original_output' in item and 'mask_mapping' in item:
            valid_data.append(item)
        else:
            print(f"Warning: Skipping incorrectly formatted data: {str(item)[:100]}...")
    
    if not valid_data:
        print("No valid data found")
        return
    
    print(f"Valid data {len(valid_data)}/{len(data)} records")
    data = valid_data
    
    # Process each QA pair
    results = []
    generated_count = 0
    max_attempts = sample_count * 10  # Maximum attempts is 10 times the target sample count
    
    with tqdm(total=sample_count, desc="Generating samples") as pbar:
        while len(results) < sample_count and generated_count < max_attempts:
            generated_count += 1
            # Randomly select a QA pair
            item = random.choice(data)
            try:
                # Generate a new random seed to ensure different replacement positions even for the same QA pair
                random.seed(random.randint(0, 1000000))
                
                malicious_qa = generate_malicious_qa(item, api_key, keep_harmful_prob)
                
                # Check if harmful ratio is within allowed range
                if abs(malicious_qa['actual_harmful_ratio'] - keep_harmful_prob) > tolerance:
                    print(f"Harmful ratio {malicious_qa['actual_harmful_ratio']:.3f} exceeds allowed range, regenerating")
                    continue
                
                # Check for duplicates with existing results
                is_duplicate = False
                for existing in results[-10:]:  # Only check the most recent 10 results for efficiency
                    if malicious_qa['malicious_output'] == existing['malicious_output']:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    results.append(malicious_qa)
                    pbar.update(1)
                    print(f"Generated: {len(results)}/{sample_count} samples")
                    print(f"Current sample - Target harmful ratio: {keep_harmful_prob:.2f}, Actual harmful ratio: {malicious_qa['actual_harmful_ratio']:.2f}")
                
            except Exception as e:
                print(f"Processing failed: {str(e)}")
                if 'original_instruction' in item:
                    print(f"Problematic item: {item['original_instruction'][:50]}...")
                continue
        
        if len(results) < sample_count:
            print(f"Warning: Only {len(results)} samples generated, target count {sample_count} not reached")
    
    # Verify final average ratio
    if results:
        actual_ratios = [item['actual_harmful_ratio'] for item in results]
        avg_ratio = sum(actual_ratios) / len(actual_ratios)
        print(f"\nStatistics:")
        print(f"Average harmful ratio: {avg_ratio:.3f}")
        print(f"Minimum harmful ratio: {min(actual_ratios):.3f}")
        print(f"Maximum harmful ratio: {max(actual_ratios):.3f}")
        if abs(avg_ratio - keep_harmful_prob) > tolerance:
            print(f"Warning: Final average harmful ratio {avg_ratio:.3f} exceeds target range {keep_harmful_prob}±{tolerance}")
    
    # Write to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Processing complete, {len(results)} samples generated, results saved to {output_file}")
    except Exception as e:
        print(f"Failed to write file: {str(e)}")
        return

# 示例使用
if __name__ == "__main__":

    API_KEY = "API_KEY"
    
    # 输入输出文件路径
    INPUT_FILE = ".\mask\counterfactual_llm_as_a_judge_with_prefix_suffix.jsonl"  
    OUTPUT_FILE = "mixed_harmful_qa_0.00_100.json"  
    KEEP_HARMFUL_PROB = 0.00  
    SAMPLE_COUNT = 100  
    TOLERANCE = 0.05  
    
    
    process_json_file(INPUT_FILE, OUTPUT_FILE, API_KEY, KEEP_HARMFUL_PROB, SAMPLE_COUNT, TOLERANCE)