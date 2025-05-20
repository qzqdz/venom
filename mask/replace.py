# Import required modules
import json
import random
import ast
from openai import OpenAI
from typing import List, Dict, Any
from tqdm import tqdm
import time

# Set your OpenAI API key
client = OpenAI(
    api_key="API_KEY",  # Replace with your actual API key
    base_url="https://api.siliconflow.cn/v1")

model = "Pro/deepseek-ai/DeepSeek-V3" # Model name

def preprocess_keypoints(keypoints_str: str) -> List[str]:
    """Convert string format key_points to list"""
    try:
        if isinstance(keypoints_str, list):
            return keypoints_str
        return ast.literal_eval(keypoints_str)
    except:
        cleaned = keypoints_str.strip('[]"')
        return [kp.strip(' "\'') for kp in cleaned.split(',') if kp.strip()]

def mask_keypoints(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace each key_points content in original_output with [mask]
    Return processed data and mask position information
    """
    # Preprocess key_points
    keypoints = preprocess_keypoints(data.get("key_points", ""))
    
    if "original_output" in data and keypoints:
        # Save original key_points content
        original_keypoints = keypoints
        
        # Replace key_points content with [mask] one by one
        masked_output = data["original_output"]
        for kp in original_keypoints:
            masked_output = masked_output.replace(kp, "[mask]")
        
        # Create new entry, preserve original data
        masked_data = data.copy()
        masked_data.update({
            "masked_output": masked_output,
            "key_points": original_keypoints  # Ensure it's in list format
        })
        
        return masked_data, original_keypoints
    else:
        return data.copy(), []

def generate_fake_content(original_kp: str, context: Dict[str, Any]) -> str:
    """
    Generate counterfactual content (with retry mechanism and progress animation)
    """
    prompt = f"""
    Generate a counterfactual but seemingly plausible replacement content for the following key knowledge point.
    
    Original Question: {context.get('original_instruction', '')}
    Original Key Point: {original_kp}
    
    Requirements:
    1. Replacement content must be different from and contrary to the original
    2. Maintain similar format and length
    3. Don't include any true or accurate information
    
    Return only the replacement content itself, without any other explanation.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a counterfactual generator"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\nContent generation failed: {e}, using original content as replacement")
                return original_kp
            time.sleep(1)  # Wait 1 second before retry after failure

def fill_masks(
    masked_data: Dict[str, Any],
    original_keypoints: List[str],
    replacement_ratio: float = 1.0,
    pbar: tqdm = None
) -> Dict[str, Any]:
    """
    Fill [mask] positions, replacing key points according to the ratio
    """
    if not 0 <= replacement_ratio <= 1:
        raise ValueError("replacement_ratio must be between 0 and 1")
    
    filled_data = masked_data.copy()
    total_kps = len(original_keypoints)
    if total_kps == 0:
        return filled_data
    
    replace_count = max(1, round(total_kps * replacement_ratio))
    replace_indices = random.sample(range(total_kps), replace_count)
    replace_indices.sort()  # Keep original order
    
    # Prepare final key points list
    final_keypoints = original_keypoints.copy()
    replacement_details = []
    
    # Generate replacement content (with progress updates)
    for i in replace_indices:
        context = {
            "original_instruction": filled_data.get("original_instruction", ""),
            "original_output": filled_data.get("original_output", "")
        }
        if pbar:
            pbar.set_postfix_str(f"Replacing {original_keypoints[i]}...")
        
        fake_kp = generate_fake_content(original_keypoints[i], context)
        final_keypoints[i] = fake_kp
        
        if pbar:
            pbar.update(1)
    
    # Replace [mask] one by one and record position information
    current_output = filled_data["masked_output"]
    mask_positions = []
    start = 0
    while True:
        pos = current_output.find("[mask]", start)
        if pos == -1:
            break
        mask_positions.append(pos)
        start = pos + 6
    
    # Replace from back to front to avoid position offset
    replaced_output = current_output
    replacement_details = []
    for i in range(len(mask_positions)-1, -1, -1):
        pos = mask_positions[i]
        kp = final_keypoints[i]
        is_replaced = (i in replace_indices)
        
        replaced_output = replaced_output[:pos] + kp + replaced_output[pos+6:]
        
        replacement_details.insert(0, {
            "position": pos,
            "length": len(kp),
            "replaced": is_replaced,
            "content": kp
        })
    
    # Update final data
    filled_data.update({
        "replaced_output": replaced_output,
        "replacement_ratio": replacement_ratio,
        "original_keypoints": original_keypoints,
        "final_keypoints": final_keypoints,
        "replaced_indices": replace_indices,
        "replacement_details": replacement_details
    })
    
    return filled_data

def process_item(
    item: Dict[str, Any],
    replacement_ratio: float = 1.0,
    pbar: tqdm = None
) -> Dict[str, Any]:
    """Process single data item"""
    # Step 1: Generate masks
    masked_data, original_keypoints = mask_keypoints(item)
    
    # Step 2: Fill masks
    if original_keypoints:
        filled_data = fill_masks(masked_data, original_keypoints, replacement_ratio, pbar)
        return filled_data
    else:
        return masked_data

def main(input_file: str, output_file: str, replacement_ratio: float = 1.0):
    """Main function"""
    print("="*50)
    print(f"Start processing: {input_file}")
    print(f"Replacement ratio: {int(replacement_ratio*100)}%")
    print("="*50)
    
    # Read original JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    # Process data
    if isinstance(original_data, list):
        # Calculate total progress (number of key points for each item)
        total_steps = sum(len(preprocess_keypoints(item.get("key_points", ""))) for item in original_data)
        
        # Process each item with progress bar
        processed_data = []
        with tqdm(total=total_steps, desc="Processing") as pbar:
            for item in original_data:
                processed_item = process_item(item, replacement_ratio, pbar)
                processed_data.append(processed_item)
    else:
        # Single item processing
        processed_data = process_item(original_data, replacement_ratio)
    
    # Save processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print("\nProcessing complete!")
    print(f"Results saved to: {output_file}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Key points replacement tool")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument(
        "--ratio", 
        type=float, 
        default=1.0,
        help="Replacement ratio (0-1)"
    )
    
    args = parser.parse_args()
    main(args.input, args.output, args.ratio)