import openai
import json
import os
import base64
import io
import threading
from datetime import datetime
from PIL import Image
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_PATHS = {
    "agentnetbench": "/path/to/project",
    "framethinker": "/path/to/project",
    "gaia": "/path/to/project",
    "gta": "/path/to/project"
}

def get_real_file_path(dataset_name, relative_path):
    """分发函数：根据数据集名称调用对应的路径解析逻辑"""
    dataset_key = dataset_name.lower()
    
    # 检查是否配置了该数据集
    if dataset_key not in ROOT_PATHS:
        return None # 对于不需要处理文件的纯文本数据集
        
    root = ROOT_PATHS[dataset_key]
    # 我把gaia改成跟其他一样的了
    return os.path.join(root, relative_path)

# ==========================================
# 1. 基础工具函数 (通用)
# ==========================================

def create_log_dir():
    """确保日志目录存在"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"Logging errors to: {LOG_FILE_PATH}")
    except Exception as e:
        print(f"CRITICAL: Could not create log directory {LOG_DIR}. Error: {e}")
        raise

def load_processed_indices(output_file):
    """
    读取已存在的结果文件
    获取已处理的 (index, source_dataset) 元组集合以便跳过
    """
    processed = set()
    if not os.path.exists(output_file):
        return processed 
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 修改点：同时获取 index 和 source_dataset
                    idx = data.get('index')
                    src = data.get('source_dataset')
                    
                    # 只有当两者都存在时才加入已处理集合
                    if idx is not None and src is not None:
                        processed.add((idx, src))
                        
                except json.JSONDecodeError:
                    continue
        if processed:
            print(f"Found {len(processed)} already processed items. They will be skipped.")
    except Exception as e:
        print(f"Warning: Could not read processed items from {output_file}. Error: {e}")
    
    return processed

def log_error(index, error_message):
    """将错误信息线程安全地写入日志文件"""
    with log_lock:
        try:
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] Index: {index} | Error: {error_message}\n")
        except Exception as e:
            print(f"CRITICAL: Failed to write to log file {LOG_FILE_PATH}. Error: {e}")

def encode_image_to_base64(image_path):
    """将图片文件编码为 Base64 字符串"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
        
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

def get_model_generation(client, system_prompt, user_content):
    """通用模型调用函数"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    max_tokens = 62678
    if "gpt-4o" in TESTED_MODEL_NAME:
        max_tokens = 16384
        
    response = client.chat.completions.create(
        model=TESTED_MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0
    )
    
    model_response_string = response.choices[0].message.content
    return model_response_string


def create_agentnetbench_system_prompt(tool_bank):
    prompt = f"""
You are an expert UI automation agent. Your task is to analyze a user's **request (query)**, a **"START" screen** (the initial state), and a **"GOAL" screen** (a visual reference of a key state).

Your primary objective is to generate a high-level "plan" and a detailed, **linear tool chain** (named "tool_chain") that **fully solves ALL tasks in the user's query**. You will **not** execute the tools or receive any feedback. You must generate the *most likely* successful path.

Use the "START" screen as your starting point.

**Crucial Assumptions:**
1.  **All Tools Work Perfectly:** Assume all tools are bug-free.
2.  **Query is Solvable:** The query can be fully solved by the tools.

**Available Tools:**
You must *only* use the low-level tools provided in the list below.
{tool_bank}

**CRITICAL: Output Format and Parameter Rules**
You must return your response *only* as a JSON object.

**1. Structure Definition:**
1.  **Root:** An object with "plan" (string) and "tool_chain" (list).
2.  **`plan`:** A high-level, natural-language summary of your strategy.
3.  **`tool_chain`:** A **list** of "Tool Call" objects.

**2. Tool Call Object Rules:**
* Each object in the `tool_chain` list **must** have 3 keys:
    * `"name"`: (string) The exact name of the tool to be called.
    * `"parameter_description"`: (object) The parameter_description for the tool.
    * `"reason"`: (string) An explanation of *why* this tool is being called at this step.

**3. parameter_description Rules (Most Important):**
* Inside the `"parameter_description"` object, you **MUST** describe the parameter values conceptually to show you understand the data flow and the *visual UI target*.
* **For static values (from the query):** Use the actual value (e.g., `"text": "Shanghai, China"`).
* **For visual targets (e.g., `target`):** Describe the UI element you are interacting with (e.g., `"target": "The 'Save' button"` or `"target": "The text box labeled 'Address'"`).
* **For dynamic values (from a previous step):** Describe *what* the data is or *where it comes from* (e.g., `"frames_to_check": "The list of frames returned from Step 1 where the girl was found"` or `"document_id": "The ID of the document created in the previous step"`).

**Example JSON Structure (for this task):**
{{
  "plan": "A high-level, natural-language summary of the sequential strategy to solve the user's query.",
  "tool_chain": [
    {{
      "name": "click",
      "parameter_description": {{ 
        "target": "The grid icon (composed of small squares) in the top-right corner" 
      }},
      "reason": "This is the first step, required to open the Google apps menu."
    }},
    {{
      "name": "click",
      "parameter_description": {{ 
        "target": "The 'Account' icon in the Google apps menu" 
      }},
      "reason": "To navigate to the main Google Account page."
    }},
    {{
      "name": "write",
      "parameter_description": {{ 
        "text": "Shanghai, China",
        "target": "The 'Address' text box after clicking 'Work'"
      }},
      "reason": "To input the work address provided in the user's query."
    }},
    {{
      "name": "click",
      "parameter_description": {{ 
        "target": "The 'Save' button" 
      }},
      "reason": "To save the newly entered address."
    }}
  ]
}}
"""
    return prompt

def build_agentnetbench_user_content(query, start_image_path, goal_image_path):
    content = []

    if not os.path.exists(start_image_path):
        raise FileNotFoundError(f"agentnetbench Required file not found: {start_image_path}")
    if not os.path.exists(goal_image_path):
        raise FileNotFoundError(f"agentnetbench Required file not found: {goal_image_path}")
    
    # 1. 添加用户 Query
    content.append({
        "type": "text",
        "text": f"Here is the user's request:\n---(Query)---\n{query}\n---(End Query)---\n"
    })
    
    # 2. 添加 Start 图像 (无 try...except)
    content.append({
        "type": "text",
        "text": "This is the STARTING screen (Initial State):"
    })
    base64_start_image = encode_image_to_base64(start_image_path)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_start_image}",
            "detail": "auto"
        }
    })
    content.append({
        "type": "text",
        "text": "This is the FINAL (GOAL) screen (Desired State):"
    })
    base64_goal_image = encode_image_to_base64(goal_image_path)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_goal_image}",
            "detail": "auto"
        }
    })

    return content

# ==========================================
# 2. Framethinker 专用逻辑
# ==========================================

def create_framethinker_system_prompt(tool_bank):
    prompt = f"""
You are an expert multimodal AI assistant. Your task is to analyze the user's query and the provided video context. Based on this, you must generate a high-level "plan" and a detailed, **linear tool chain** (named "tool_chain").

Your goal is to generate a *single, logical sequence of tool calls* that solves the query. You will **not** execute the tools or receive any feedback. You must generate the *most likely* successful path based on your own generated plan.

**Crucial Assumptions:**
1.  **All Tools Work Perfectly:** Assume all tools are bug-free.
2.  **Query is Solvable:** The query can be fully solved by the tools.

**Available Tools:**
You must *only* use the tools provided in the list below.
{tool_bank}

**CRITICAL: Output Format and Parameter_description Rules**
You must return your response *only* as a JSON object.

**1. Structure Definition:**
1.  **Root:** An object with "plan" (string) and "tool_chain" (list).
2.  **`plan`:** A high-level, natural-language summary of your strategy.
3.  **`tool_chain`:** A **list** of "Tool Call" objects.

**2. Tool Call Object Rules:**
* Each object in the `tool_chain` list **must** have 3 keys:
    * `"name"`: (string) The exact name of the tool to be called.
    * `"parameter_description"`: (object) The parameter_description for the tool.
    * `"reason"`: (string) An explanation of *why* this tool is being called at this step.

**3. parameter_description Rules (Most Important):**
* Inside the `"parameter_description"` object, you **MUST** describe the parameter values conceptually to show you understand the data flow.
    * **For static values (from the query):** Use the actual value (e.g., `"target": "girl character"`).
    * **For dynamic values (from a previous step):** Describe *what* the data is or *where it comes from* (e.g., `"frames_to_check": "The list of frames returned from Step 1 where the girl was found"` or `"document_id": "The ID of the document created in the previous step"`).

**Example JSON Structure:**
{{
  "plan": "A high-level, natural-language summary of the sequential strategy to solve the user's query.",
  "tool_chain": [
    {{
      "name": "first_tool_name",
      "parameter_description": {{
        "param_key_1": "A static value from the user query",
        "param_key_2": "Another static value"
      }},
      "reason": "Explain *why* this is the first logical step to take."
    }},
    {{
      "name": "second_tool_name",
      "parameter_description": {{
        "input_data": "The output (e.g., 'list of items') from the 'first_tool_name' call",
        "another_param": "A static value"
      }},
      "reason": "Why this tool is called *sequentially* after the first, using its output."
    }},
    {{
      "name": "final_tool_name",
      "parameter_description": {{
        "data_to_process": "The processed data from the 'second_tool_name' call"
      }},
      "reason": "Why this tool is called to produce the final answer."
    }}
  ]
}}
"""
    return prompt

def build_framethinker_user_content(query, image_paths):
    """Framethinker 专用: 按照 <image> 标签分割 Query 并插入图片"""
    query_parts = query.split("<image>")
    content = []
    
    image_index = 0
    for i, part in enumerate(query_parts):
        if part:
            content.append({"type": "text", "text": part})
        
        if i < len(query_parts) - 1:
            if image_index < len(image_paths):
                img_path = image_paths[image_index]
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Framethinker Required file not found: {img_path}")
                base64_image = encode_image_to_base64(img_path)
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto"
                    }
                })
                image_index += 1
            else:
                content.append({
                    "type": "text", 
                    "text": f"\n[Warning: Missing image data for <image> tag {i+1}]"
                })
                
    return content

# ==========================================
# 3. GAIA 专用逻辑
# ==========================================

def create_gaia_system_prompt(tool_bank):
    prompt = f"""
You are an expert AI assistant. Your task is to analyze the user's query and any provided files (which could be text or images). Based on this, you must generate a high-level "plan" and a detailed, **linear tool chain** (named "tool_chain").

Your goal is to generate a *single, logical sequence of tool calls* that solves the query. You will **not** execute the tools or receive any feedback. You must generate the *most likely* successful path based on your own generated plan.

**Crucial Assumptions:**
1.  **All Tools Work Perfectly:** Assume all tools are bug-free.
2.  **Query is Solvable:** The query can be fully solved by the tools.

**Available Tools:**
You must *only* use the tools provided in the list below.
{tool_bank}

**CRITICAL: Output Format and Parameter_description Rules**
You must return your response *only* as a JSON object.

**1. Structure Definition:**
1.  **Root:** An object with "plan" (string) and "tool_chain" (list).
2.  **`plan`:** A high-level, natural-language summary of your strategy.
3.  **`tool_chain`:** A **list** of "Tool Call" objects.

**2. Tool Call Object Rules:**
* Each object in the `tool_chain` list **must** have 3 keys:
    * `"name"`: (string) The exact name of the tool to be called.
    * `"parameter_description"`: (object) The parameter_description for the tool.
    * `"reason"`: (string) An explanation of *why* this tool is being called at this step.

**3. parameter_description Rules (Most Important):**
* Inside the `"parameter_description"` object, you **MUST** describe the parameter values conceptually to show you understand the data flow.
    * **For static values (from the query):** Use the actual value (e.g., `"target": "girl character"`).
    * **For dynamic values (from a previous step):** Describe *what* the data is or *where it comes from* (e.g., `"frames_to_check": "The list of frames returned from Step 1 where the girl was found"` or `"document_id": "The ID of the document created in the previous step"`).

**Example JSON Structure:**
{{
  "plan": "A high-level, natural-language summary of the sequential strategy to solve the user's query.",
  "tool_chain": [
    {{
      "name": "first_tool_name",
      "parameter_description": {{
        "param_key_1": "A static value from the user query",
        "param_key_2": "Another static value"
      }},
      "reason": "Explain *why* this is the first logical step to take."
    }},
    {{
      "name": "second_tool_name",
      "parameter_description": {{
        "input_data": "The output (e.g., 'list of items') from the 'first_tool_name' call",
        "another_param": "A static value"
      }},
      "reason": "Why this tool is called *sequentially* after the first, using its output."
    }},
    {{
      "name": "final_tool_name",
      "parameter_description": {{
        "data_to_process": "The processed data from the 'second_tool_name' call"
      }},
      "reason": "Why this tool is called to produce the final answer."
    }}
  ]
}}
"""
    return prompt

def build_gaia_user_content(query, files_list):
    """GAIA 专用: 附加文件内容 (txt) 或图片 (base64) 到 Query 之后"""
    content = []
    content.append({"type": "text", "text": query})
    
    if not files_list:
        return content

    try:
        file_info = files_list[0]
        file_path = file_info['path']
        full_path = get_real_file_path('gaia', file_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"GAIA Required file not found: {full_path}")

        low_case_path = file_path.lower()
        
        if low_case_path.endswith('.txt'):
            with open(full_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            content.append({
                "type": "text",
                "text": f"--- Start of provided file content: {file_path} ---\n{file_content}\n--- End of provided file content ---"
            })
        
        elif low_case_path.endswith('.jpg') or low_case_path.endswith('.png') or low_case_path.endswith('.jpeg'):
            try: 
                base64_image = encode_image_to_base64(full_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}", 
                        "detail": "auto"
                    }
                })
            except Exception as e:
                print("gaia failed to encode image")
                raise
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Only .txt, .jpg, .png, .jpeg are supported.")
    
    except Exception as e:
        print(f"CRITICAL: Failed to process file {full_path}. Error: {e}")
        raise 

    return content



def create_gta_system_prompt(tool_bank):
    prompt = f"""
You are an expert multimodal AI assistant. Your task is to analyze the user's query and the provided images. Based on this, you must generate a high-level "plan" and a detailed, **linear tool chain** (named "tool_chain").

Your goal is to generate a *single, logical sequence of tool calls* that solves the query. You will **not** execute the tools or receive any feedback. You must generate the *most likely* successful path based on your own generated plan.

**Crucial Assumptions:**
1.  **All Tools Work Perfectly:** Assume all tools are bug-free.
2.  **Query is Solvable:** The query can be fully solved by the tools.

**Available Tools:**
You must *only* use the tools provided in the list below.
{tool_bank}

**CRITICAL: Output Format and Parameter_description Rules**
You must return your response *only* as a JSON object.

**1. Structure Definition:**
1.  **Root:** An object with "plan" (string) and "tool_chain" (list).
2.  **`plan`:** A high-level, natural-language summary of your strategy.
3.  **`tool_chain`:** A **list** of "Tool Call" objects.

**2. Tool Call Object Rules:**
* Each object in the `tool_chain` list **must** have 3 keys:
    * `"name"`: (string) The exact name of the tool to be called.
    * `"parameter_description"`: (object) The parameter_description for the tool.
    * `"reason"`: (string) An explanation of *why* this tool is being called at this step.

**3. parameter_description Rules (Most Important):**
* Inside the `"parameter_description"` object, you **MUST** describe the parameter values conceptually to show you understand the data flow.
    * **For static values (from the query):** Use the actual value (e.g., `"target": "girl character"`).
    * **For dynamic values (from a previous step):** Describe *what* the data is or *where it comes from* (e.g., `"frames_to_check": "The list of frames returned from Step 1 where the girl was found"` or `"document_id": "The ID of the document created in the previous step"`).

**4. STRICT Data Access Rule:**
* DO NOT incorporate any direct, specific visual information from the image (e.g., *exact* text content, *actual* count of objects/people, specific object coordinates) into the `"plan"`, `"parameter_description"`, or `"reason"` fields. You should call an appropriate tool first to acquire this information.

**Example JSON Structure:**
{{
  "plan": "A high-level, natural-language summary of the sequential strategy to solve the user's query.",
  "tool_chain": [
    {{
      "name": "first_tool_name",
      "parameter_description": {{
        "param_key_1": "A static value from the user query",
        "param_key_2": "Another static value"
      }},
      "reason": "Explain *why* this is the first logical step to take."
    }},
    {{
      "name": "second_tool_name",
      "parameter_description": {{
        "input_data": "The output (e.g., 'list of items') from the 'first_tool_name' call",
        "another_param": "A static value"
      }},
      "reason": "Why this tool is called *sequentially* after the first, using its output."
    }},
    {{
      "name": "final_tool_name",
      "parameter_description": {{
        "data_to_process": "The processed data from the 'second_tool_name' call"
      }},
      "reason": "Why this tool is called to produce the final answer."
    }}
  ]
}}
"""
    return prompt

def build_gta_user_content(query, image_paths):
    """
    (新版) 将 query 文本和图片路径列表构建为 OpenAI API 需要的 content 列表。
    此版本将 query 文本放在最前面，然后按顺序附加所有图片。
    """
    content = []
    
    # 1. (修改) 首先添加完整的、未经分割的 query 文本
    content.append({"type": "text", "text": query})
    
    # 2. (修改) 遍历所有 image_paths，编码并附加它们
    for img_path in image_paths:
        try:
            # encode_image_to_base64 会在图片路径不存在或无法打开时抛出异常
            base64_image = encode_image_to_base64(img_path)
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "auto" # 您可以根据需要将其更改为 "high"
                }
            })
        except Exception as e:
            # (重要) 遵循您的要求：如果任何一张图片读取失败，立即抛出异常
            print(f"CRITICAL: Failed to encode image {img_path}. Error: {e}")
            raise # 重新抛出异常，以便 process_line 可以捕获它

    return content

def create_skywork_system_prompt(tool_bank, task_specific_prompt):
    prompt_first_part = f"""
**Background Context:**
Here is the background context for the task:
{task_specific_prompt}

**Your Task (Tool Chain Generation):**
Based on the user's query, you must generate a high-level "plan" and a detailed, **linear tool chain** (named "tool_chain"). You will **not** execute the tools or receive any feedback. Your goal is to generate a *single, logical sequence of steps* that correctly solves the query. When generating the plan and toolchain, you need to comply with the requirements and constraints of the Background Context. In all circumstances, your answer must be in English.

**Crucial Assumptions:**
1.  **All Tools Work Perfectly:** Assume all tools are bug-free.
2.  **Query is Solvable:** The query can be fully solved by the tools.
"""
    
    prompt_tool_part = f"""

    **Available Tools:**
    You must *only* use the tools provided in the list below.
    {tool_bank}
    If the AVAILABLE Tools is empty, it means that all available tools are in the Background Context.
    """

    prompt_last_part = f"""
**CRITICAL: Output Format and Parameter_description Rules**
You must return your response *only* as a JSON object.

**1. Structure Definition:**
1.  **Root:** An object with "plan" (string) and "tool_chain" (list).
2.  **`plan`:** A high-level, natural-language summary of your strategy.
3.  **`tool_chain`:** A **list** of "Tool Call" objects.

**2. Tool Call Object Rules:**
* Each object in the `tool_chain` list **must** have 3 keys:
    * `"name"`: (string) The exact name of the tool to be called.
    * `"parameter_description"`: (object) The parameter_description for the tool.
    * `"reason"`: (string) An explanation of *why* this tool is being called at this step.

**3. parameter_description Rules (Most Important):**
* Inside the `"parameter_description"` object, you **MUST** describe the parameter values conceptually to show you understand the data flow.
    * **For static values (from the query):** Use the actual value (e.g., `"target": "girl character"`).
    * **For dynamic values (from a previous step):** Describe *what* the data is or *where it comes from* (e.g., `"frames_to_check": "The list of frames returned from Step 1 where the girl was found"` or `"document_id": "The ID of the document created in the previous step"`).

**Example JSON Structure:**
{{
  "plan": "A high-level, natural-language summary of the sequential strategy to solve the user's query.",
  "tool_chain": [
    {{
      "name": "first_tool_name",
      "parameter_description": {{
        "param_key_1": "A static value from the user query",
        "param_key_2": "Another static value"
      }},
      "reason": "Explain *why* this is the first logical step to take."
    }},
    {{
      "name": "second_tool_name",
      "parameter_description": {{
        "input_data": "The output (e.g., 'list of items') from the 'first_tool_name' call",
        "another_param": "A static value"
      }},
      "reason": "Why this tool is called *sequentially* after the first, using its output."
    }},
    {{
      "name": "final_tool_name",
      "parameter_description": {{
        "data_to_process": "The processed data from the 'second_tool_name' call"
      }},
      "reason": "Why this tool is called to produce the final answer."
    }}
  ]
}}
"""
    if not tool_bank:
        prompt = prompt_first_part + prompt_last_part
    else:
        prompt = prompt_first_part + prompt_tool_part + prompt_last_part
    return prompt


def build_skywork_user_content(query):
    return [{"type": "text", "text": query}]


def create_tool_bench_system_prompt(tool_bank_string, task_specific_prompt):
    system_prompt = f"""
**Background Context:**
Here is the background context for the task:
{task_specific_prompt}

**Your Task (Tool Chain Generation):**
Based on the user's query, you must generate a high-level "plan" and a detailed, **linear tool chain** (named "tool_chain"). You will **not** execute the tools or receive any feedback. Your goal is to generate a *single, logical sequence of steps* that correctly solves the query. In all circumstances, your answer must be in English.

**Crucial Assumptions:**
1.  **All Tools Work Perfectly:** Assume all tools are bug-free.
2.  **Query is Solvable:** The query can be fully solved by the tools.

**Available Tools:**
You must *only* use the tools provided in the JSON list below.
{tool_bank_string}

**CRITICAL: Output Format and parameter_description Rules**
You must return your response *only* as a JSON object.

**1. Structure Definition:**
1.  **Root:** An object with "plan" (string) and "tool_chain" (list).
2.  **`plan`:** A high-level, natural-language summary of your strategy.
3.  **`tool_chain`:** A **list** of "Tool Call" objects.

**2. Tool Call Object Rules:**
* Each object in the `tool_chain` list **must** have 3 keys:
    * `"name"`: (string) The exact name of the tool to be called (e.g., "api_name" or "tool_name").
    * `"parameter_description"`: (object) The parameter_description for the tool.
    * `"reason"`: (string) An explanation of *why* this tool is being called at this step.

**3. parameter_description Rules (Most Important):**
* Inside the `"parameter_description"` object, you **MUST** describe the parameter values conceptually to show you understand the data flow.
    * **For static values (from the query):** Use the actual value (e.g., `"target": "girl character"`).
    * **For dynamic values (from a previous step):** Describe *what* the data is or *where it comes from* (e.g., `"frames_to_check": "The list of frames returned from Step 1 where the girl was found"` or `"document_id": "The ID of the document created in the previous step"`).

**Example JSON Structure:**
{{
  "plan": "A high-level, natural-language summary of the sequential strategy to solve the user's query.",
  "tool_chain": [
    {{
      "name": "first_tool_name",
      "parameter_description": {{
        "param_key_1": "A static value from the user query",
        "param_key_2": "Another static value"
      }},
      "reason": "Explain *why* this is the first logical step to take."
    }},
    {{
      "name": "second_tool_name",
      "parameter_description": {{
        "input_data": "The output (e.g., 'list of items') from the 'first_tool_name' call",
        "another_param": "A static value"
      }},
      "reason": "Why this tool is called *sequentially* after the first, using its output."
    }}
  ]
}}
"""
    return system_prompt


# ==========================================
# 4. 主处理逻辑 (合并版)
# ==========================================

def process_line(line, client):
    """
    根据 source_dataset 自动选择处理逻辑
    """
    data = None
    index = "UNKNOWN"
    try:
        data = json.loads(line.strip())
        index = data.get('index', 'UNKNOWN')
        source_dataset = data.get('source_dataset', 'unknown').lower()

        query = data['query']
        tool_bank = data['background']['tool_bank']
        
        # === 核心分支逻辑 ===
        system_prompt = ""
        user_content = []
        
        if "agentnetbench" in source_dataset:
            # 使用 agentnetbench 逻辑
            image_paths = [file_info['path'] for file_info in data['background'].get('files', [])]
            image_paths = [get_real_file_path('agentnetbench', file_path) for file_path in image_paths]
            system_prompt = create_agentnetbench_system_prompt(tool_bank)
            user_content = build_agentnetbench_user_content(query, image_paths[0], image_paths[1])

        elif "framethinker" in source_dataset:
            # 使用 Framethinker 逻辑
            image_paths = [file_info['path'] for file_info in data['background'].get('files', [])]
            image_paths = [get_real_file_path('framethinker', file_path) for file_path in image_paths]
            system_prompt = create_framethinker_system_prompt(tool_bank)
            user_content = build_framethinker_user_content(query, image_paths)
            
        elif "gaia" in source_dataset:
            # 使用 GAIA 逻辑
            files_list = data['background'].get('files', [])
            system_prompt = create_gaia_system_prompt(tool_bank)
            user_content = build_gaia_user_content(query, files_list)
        
        elif "gta" in source_dataset:
            # 使用 gta 逻辑
            image_paths = [file_info['path'] for file_info in data['background'].get('files', [])]
            image_paths = [get_real_file_path('gta', file_path) for file_path in image_paths]
            system_prompt = create_gta_system_prompt(tool_bank)
            user_content = build_gta_user_content(query, image_paths)
        
        elif "skywork" in source_dataset:
            task_specific_prompt = data['system_prompt'] 
            system_prompt = create_skywork_system_prompt(tool_bank, task_specific_prompt)
            user_content = build_skywork_user_content(query)
        
        elif "tool_bench" in source_dataset:
            task_specific_prompt = """You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Then you will analyze your status now, then decide what to do next."""
            system_prompt = create_tool_bench_system_prompt(tool_bank, task_specific_prompt)
            user_content = [{"type": "text", "text": query}]
            
        else:
            # 未知数据集，抛出错误
            raise ValueError(f"Unknown source_dataset: {source_dataset}")

        # === 调用模型 ===
        model_response_string = get_model_generation(
            client,
            system_prompt, 
            user_content
        )
        
        # === 构建输出 ===
        result = {
            "index": data['index'],
            "source_dataset": data.get('source_dataset'), # 保留 source_dataset
            "steps": data['steps'],
            "tested_model": TESTED_MODEL_NAME,
            "query": data['query'],
            "ground_truth_plan": data['plan'],
            "ground_truth_tool_chain": data['tool_chain'],
            "available_tools": data['background']['tool_bank'],
            "model_response": model_response_string,
            "test_system_prompt": system_prompt,
            "background": data['background']
        }
        
        return result
        
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        print(f"ERROR on index {index} (Source: {data.get('source_dataset') if data else 'N/A'}): {error_message}. Skipping.")
        log_error(index, error_message)
        return None


def run_evaluation(client, data_num=-1):
    create_log_dir()
    
    # 加载已处理的 (index, source_dataset) 集合
    processed_keys = load_processed_indices(OUTPUT_JSONL_FILE)
    
    print(f"Starting evaluation...\nModel: {TESTED_MODEL_NAME}\nInput: {INPUT_JSONL_FILE}\nOutput: {OUTPUT_JSONL_FILE}")
    
    lines_to_process = []
    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as infile:
            line_iterator = infile
            if data_num > 0:
                line_iterator = itertools.islice(infile, data_num)
            for line in line_iterator:
                try:
                    data = json.loads(line)
                    
                    # 修改点：构建当前数据的联合键 (index, source_dataset)
                    current_idx = data.get('index')
                    current_src = data.get('source_dataset')
                    current_key = (current_idx, current_src)
                    
                    # 检查联合键是否已存在
                    if current_key not in processed_keys:
                        lines_to_process.append(line)
                        
                except json.JSONDecodeError:
                    print(f"Skipping corrupted line in input file: {line[:50]}...")
    except FileNotFoundError:
        print(f"CRITICAL: Input file not found at {INPUT_JSONL_FILE}")
        return
    except Exception as e:
        print(f"CRITICAL: Failed to read input file. Error: {e}")
        return

    total_to_process = len(lines_to_process)
    if total_to_process == 0:
        print("No new items to process. All items already exist in the output file.")
        return
        
    print(f"Total items to process: {total_to_process} (Skipped {len(processed_keys)} existing items)")
    
    with open(OUTPUT_JSONL_FILE, 'a', encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
            
            future_to_line = {
                executor.submit(process_line, line, client): line 
                for line in lines_to_process
            }
            
            processed_count = 0
            for future in as_completed(future_to_line):
                result = future.result()
                
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()
                
                processed_count += 1
                print(f"Progress: {processed_count}/{total_to_process} ({(processed_count/total_to_process)*100:.1f}%)")

    print(f"Evaluation complete. Results appended to {OUTPUT_JSONL_FILE}")
    print(f"Check {LOG_FILE_PATH} for any errors.")

# ==========================================
# 5. 执行配置
# ==========================================
if __name__ == "__main__":
    
    # === 定义代理地址 ===
    PROXY_URL = "http://YOUR_PROXY"
    
    # === 定义模型配置 ===
    MODEL_CONFIGS = {
        "step-3.5-flash": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "gemini": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "gpt-4o": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "gpt-5": {
            # "api_key": "YOUR_API_KEY",
            # "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "claude": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "qwen3vl-235B": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "qwen3vl-30B": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "internvl3-5-241B": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "internvl3-5-30B": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        }
    }

    # === 全局路径设置 ===
    INPUT_JSONL_FILE = '/path/to/project'
    LOG_DIR = '/path/to/project'
    
    data_num = -1
    MAX_CONCURRENT_CALLS = 3

    # === 测试模型列表 ===
    TESTED_MODEL_NAME_LIST = [
        # "gemini-2.5-pro",
        # "gemini-2.5-flash",
        # "gpt-4o",
        # "gpt-5",
        # "claude-sonnet-4-5-20250929",
        # "qwen3vl-235B",
        # "qwen3vl-30B",
        # "internvl3-5-241B",
        # "internvl3-5-30B",
        "step-3.5-flash",
    ]

    for model_name in TESTED_MODEL_NAME_LIST:
        print("\n" + "="*80)
        print(f"STARTING TEST FOR MODEL: {model_name}")
        print("="*80 + "\n")
        
        # 1. 匹配 Config
        config_key = "default" 
        if "gemini" in model_name:
            config_key = "gemini"
        elif "gpt-4o" in model_name:
            config_key = "gpt-4o"
        elif "gpt-5" in model_name:
            config_key = "gpt-5"
        elif "claude" in model_name:
            config_key = "claude"
        elif "qw3_2050_agent_planner" in model_name:
            config_key = "qw3_2050_agent_planner"
        elif "qwen3vl-235B" in model_name:
            config_key = "qwen3vl-235B"
        elif "qwen3vl-30B" in model_name:
            config_key = "qwen3vl-30B"
        elif "internvl3-5-241B" in model_name:
            config_key = "internvl3-5-241B"
        elif "internvl3-5-30B" in model_name:
            config_key = "internvl3-5-30B"
        elif "step-3.5-flash" in model_name:
            config_key = "step-3.5-flash"
            
        try:
            config = MODEL_CONFIGS[config_key]
            print(f"Using config for '{config_key}'...")
        except KeyError:
            print(f"Warning: No config found for '{config_key}'. Using 'default' config.")
            config = MODEL_CONFIGS["default"]

        # 2. 设置代理
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
        
        if config_key == 'qwen3vl-235B' or config_key == 'qwen3vl-30B' or config_key == 'internvl3-5-241B' or config_key == 'internvl3-5-30B':
            print("  -> Mode: Local (Config: default). Disabling Proxy.")
            for var in proxy_vars:
                if var in os.environ:
                    del os.environ[var]

        # 3. 创建 Client
        client = openai.OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )

        # 4. 设置当前测试的全局变量
        TESTED_MODEL_NAME = model_name
        
        # if '/' in model_name:
        #     model_name_for_file = model_name.split('/')[-1]
        if model_name == 'stepfun/step-3.5-flash:free':
            model_name_for_file = 'step-3.5-flash'
        elif model_name == 'claude-sonnet-4-5-20250929':
            model_name_for_file = 'claude-sonnet-4-5'
        else:
            model_name_for_file = model_name
            
        # 根据要求修改输出文件名格式: {model_name}_efficiency.jsonl
        OUTPUT_JSONL_FILE = f'{LOG_DIR}/{model_name_for_file}_offline.jsonl' 
        
        LOG_FILE_PATH = os.path.join(LOG_DIR, f"{model_name_for_file}_test.log")
        log_lock = threading.Lock()

        # 5. 运行评测
        run_evaluation(client, data_num)

        print(f"\n--- COMPLETED TEST FOR MODEL: {model_name} ---")

    print("\n" + "="*80)
    print("All model evaluations finished.")
