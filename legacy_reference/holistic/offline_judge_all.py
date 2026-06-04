import openai
import json
import os
import io
import base64
import threading
import itertools
import re
from datetime import datetime
from PIL import Image
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
# 0. 基础辅助工具 (Shared Utilities)
# ==========================================

def encode_image_to_base64(image_path):
    """
    通用图片编码函数。
    尝试将图片转换为Base64。如果失败，打印错误并返回 None。
    具体的错误处理（是忽略还是报错停止）由调用方决定。
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return None
        
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Warning: Failed to encode image {image_path}. Error: {e}")
        return None

def create_agentnetbench_system_prompt():
    evaluation_philosophy = """
    **Your Evaluation Philosophy (Crucial):**
    You are evaluating a UI Agent's plan. Your goal is to determine if the plan is **functionally viable** to solve the user's intent, NOT if it perfectly mimics the expert.

    1. **"Viability" over "Pedantry":** UI automation is flexible. 
        - **Implicit Actions:** Assume high-level tools handle basic preconditions. E.g., `write` usually implies focusing the field first. `dragto` usually implies selecting the object first. Do NOT flag missing "clicks" before "writes" as errors unless the UI specifically requires a complex expansion (like opening a closed dropdown).
        - **Shortcut Tolerance:** Using hotkeys (e.g., `Ctrl+H`) instead of UI menus is **Excellent**, not an error.
        - **Browser Features:** Interacting with browser native features (like clicking a "Download Complete" chip) is valid.

    2. **Handling "Task Tainting" (Nuance is Key):**
        - **FATAL (E1 Error):** The model ignores the *calculation* or *process* and hardcodes the *final answer* found in the Goal Screen (e.g., typing "42" instead of using a calculator).
        - **ACCEPTABLE (No Error):** The model adds harmless cosmetic steps to visually match the Goal Screen *after* the main task is done (e.g., closing a window, showing the desktop, scrolling to a specific view). This is "Diligent", not "Tainted".

    3. **Alternative Paths are Valid:** The `expert_example` is just ONE way to solve it. 
        - If the model searches "Song + Artist" directly, while the expert searches "Artist" then clicks "Song", the model is **CORRECT** (and likely more efficient).
        - Do NOT penalize efficient deviations.
    """

    evaluation_rubric = """
    **Evaluation Rubric: 6-Category Error Catalog**

    **1. E1_GOAL_UNDERSTANDING (Intent Mismatch):**
    - Model solves the wrong problem.
    - Model **Hardcodes the result** (cheating).
    - *Note: Harmless cosmetic steps (like aligning the view) are NOT E1.*

    **2. E2_TASK_COMPLETENESS:**
        - Misses a sub-task (e.g., "Download and Rename" -> only Downloads).

    **3. E3_CONSTRAINT_VIOLATION:**
        - Violates negative constraints (e.g., "Don't use mouse").

    **4. E4_LOGICAL_DEFECT (The "Crash" Test):**
        - Mark as Error ONLY if the sequence would **fail/crash** or leads to a dead end.
        - **Do NOT mark as Error:** inefficient steps, redundant clicks, or different navigation paths (e.g., Search vs. Menu).
        - **Do NOT mark as Error:** missing atomic clicks (e.g., assume `write` works on the target text box).
        - **Example of Error:** Clicking "Save" on a modal that hasn't been opened yet.

    **5. E5_TOOL_USE_ERROR:**
        - Passing wrong parameter types.
        - Using a tool for the wrong purpose (e.g., using `moveto` to "verify" text).

    **6. E6_HALLUCINATION_ERROR:**
        - Inventing UI elements that likely do not exist in standard apps (e.g., a "Calculate Cost" button in a simple Notepad app).
        - *Be careful:* Standard shortcuts (Ctrl+A, Ctrl+C) exist almost everywhere.
    """
    
    grading_instruction = """
    **Evaluation Protocol (Mental Simulation):**
    Before generating the JSON, you must perform a mental simulation:
    1. **Simulate:** Imagine executing the `model_response` step-by-step on the `START` screen.
    2. **Ask:** "Would this action crash? Would it fail to input text?" (Be generous: assume standard UI responsiveness).
    3. **Compare Constraints:** Did the user forbid this method?
    4. **Verdict:** If it works, it works. Grade 1.0 even if it looks different from the expert.
    """

    grading_rubric = """
    **Grading Scale:**
    - **1.0 (Correct):** Plan is logically sound and solves the query. (Alternative paths included).
    - **0.8 (Minor Flaw):** Solves the task but has a tiny inefficiency or harmless redundancy.
    - **0.6 (Mostly Correct):** Gets the main flow right but fails a specific sub-task or has a logical break.
    - **0.0 - 0.4:** Completely wrong or hallucinated.
    """
    
    system_prompt = f"""
    You are an expert, pragmatic AI Agent Evaluator. 
    You focus on **Solvability**, not **Mimicry**.

    {evaluation_philosophy}
    
    {evaluation_rubric}
    
    {grading_rubric}
    
    {grading_instruction}

    **Output Format Requirement:**
    You must return your response *only* as a JSON object, without any explanatory text or markdown.
    The JSON object must have the following **four** keys: `is_correct`, `grade`, `error_list`, and `reasoning`.

    * `is_correct`: (boolean) `true` if `error_list` is all zeros, `false` otherwise.
    * `grade`: (float) A score from `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, following the grading rubric.
    * `error_list`: (array) A 6-element array corresponding to errors [E1, E2, E3, E4, E5, E6]. The tested plan can be judged to have made multiple errors simultaneously.
    * `reasoning`: (string) A short concise explanation for the `error_list` and the assigned `grade`.

    **Example Output:**
    {{
      "is_correct": boolean,
      "grade": float,
      "error_list": [E1, E2, E3, E4, E5, E6],
      "reasoning": "Your reason for making this judgment."
    }}
    """
    return system_prompt

def create_agentnetbench_user_content(data):
    generator_system_prompt = data.get('test_system_prompt', 'ERROR: test_system_prompt NOT FOUND IN DATA')
    
    # 2. 格式化 GT
    gt_chain = json.dumps(data.get('ground_truth_tool_chain', 'N/A'), indent=2, ensure_ascii=False)
    
    # 3. 格式化模型回复
    model_resp_str = data['model_response']
    if isinstance(model_resp_str, dict):
        model_resp_str = json.dumps(model_resp_str, indent=2, ensure_ascii=False)
    else:
        try:
            # 尝试解析，使其格式更美观，如果失败也无妨
            parsed_resp = json.loads(model_resp_str)
            model_resp_str = json.dumps(parsed_resp, indent=2, ensure_ascii=False)
        except Exception:
            pass # 如果失败，就用原始字符串

    text_content = f"""
    Here is the case to evaluate. Please provide your judgement based on the system prompt's rubric.

    --- 1. USER QUERY ---
    {data['query']}

    --- 2. GENERATOR'S SYSTEM PROMPT (Contains Tool Definitions) ---
    This is the *exact* prompt the model saw. Use it to understand the task's goals and constraints, and find the tool definitions within this text.
    {generator_system_prompt}

    --- 3. EXPERT EXAMPLE (FOR REFERENCE ONLY) ---
    * **Tool Chain:** {gt_chain}

    --- 4. MODEL'S GENERATED PLAN (TO BE JUDGED) ---
    {model_resp_str}
    """
    
    # --- (修改) 增加图片处理逻辑 ---
    # 1. 提取图片路径
    if 'background' not in data or 'files' not in data['background']:
        raise ValueError("Background files information missing in data.")
        
    image_paths = [file_info['path'] for file_info in data['background']['files']]
    image_paths = [get_real_file_path('agentnetbench', img_path) for img_path in image_paths]
    
    if len(image_paths) < 2:
        raise ValueError(f"Data requires at least 2 images (start and goal), but found {len(image_paths)}")
        
    start_image_path = image_paths[0]
    goal_image_path = image_paths[1]
    
    # 2. 编码图片 (如果文件不存在，会在此处抛出异常，外层会捕获)
    base64_start = encode_image_to_base64(start_image_path)
    base64_goal = encode_image_to_base64(goal_image_path)

    # 3. 构建 Multimodal 消息列表
    content = []
    
    # 添加 Start Image
    content.append({
        "type": "text",
        "text": "This is the STARTING screen (Initial State):"
    })
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_start}",
            "detail": "auto"
        }
    })
    
    # 添加 Goal Image
    content.append({
        "type": "text",
        "text": "This is the FINAL (GOAL) screen (Desired State):"
    })
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_goal}",
            "detail": "auto"
        }
    })
    
    # 添加文本内容 (Query, System Prompt, GT, Model Plan)
    content.append({
        "type": "text",
        "text": text_content
    })

    return content


def create_framethinker_system_prompt():
    evaluation_philosophy = """
    **Your Evaluation Philosophy (Crucial):**
    You are an expert logical verifier. You must determine if the `model_response` is **logically capable of solving the `user_query`**.

    **1. The "Smart Runtime" Assumption (Permissiveness on Data Types):**
       - Do NOT penalize the model for minor data structure mismatches (e.g., Tool A returns a JSON object containing a list, but Tool B's input expects a raw list).
       - Assume an intelligent Agent Runtime exists that can parse/extract the correct fields from Step A to feed into Step B.
       - **Only flag E4 (Logical Defect)** if the data is *fundamentally unavailable* (e.g., trying to analyze "audio" from a tool that only returned "bounding boxes").

    **2. Handling "Simulated Execution" (The "Oracle" Problem):**
       - Models often hallucinate the *final answer* in the `parameter_description` of the last step (e.g., `output(answer="The girl is sad...")`).
       - **Standardization:** Do **NOT** mark this as E4 (Logic) or E6 (Hallucination). Mark this specifically as **E3 (Constraint Violation - Format)** because the plan logic itself might still be valid.
       - **Grading Impact:** If a plan is perfect except for this "Simulated Execution" error, the grade must be **0.8**, not lower.

    **3. Strict Constraint Verification:**
       - You must strictly verify explicit constraints in the `user_query` (e.g., "Only use frames 0-100", "Do not use OCR").
    """

    evaluation_rubric = """
    **Evaluation Rubric Part 1: 6-Category Error Catalog**
    Set 1 (Error Present) or 0 (Error Absent).

    **1. E1_GOAL_UNDERSTANDING:**
       - Misunderstands the core intent (e.g., asks for "summary", plan gives "count").

    **2. E2_TASK_COMPLETENESS:**
       - Misses sub-tasks (e.g., Query asks for A, B, and C; plan only solves A and B).

    **3. E3_CONSTRAINT_VIOLATION & FORMAT:**
       - **Scope Violation:** Uses frames outside requested range (e.g., Query: "Frames 0-100", Plan: "Frame 200").
       - **Simulated Execution:** The model writes a specific, hallucinated answer in the tool parameters (e.g., `answer="The car is red"` before the tool runs).
       - **Formatting:** Fails to describe parameters conceptually (as required by system prompt).

    **4. E4_LOGICAL_DEFECT (The Flow Killer):**
       - **Broken Dependency:** Step 2 requires data that Step 1 *never* generates (even with a Smart Runtime).
       - **Circular Logic:** Step A depends on Step B, but Step B depends on Step A.
       - **Impossible Action:** Trying to "zoom in" without a cropping tool; trying to "read text" from a tool that only detects colors.

    **5. E5_TOOL_USE_ERROR:**
       - **Definition:** Fundamentally using the wrong tool for the job.
       - **Exemption:** Do NOT flag if the parameter *value* is merely suboptimal (tuning). Flag only if the *concept* is wrong (e.g., passing an Image ID to a `TextSummarizer`).

    **6. E6_HALLUCINATION_ERROR:**
       - **Fake Tools:** Calls a tool name not in `Available Tools`.
       - **Factual Hallucination (Intermediate):** Assumes specific facts in *intermediate* steps that determine the control flow (e.g., `if (tool_output == "cat")` when the output is unknown).
       - *Note: Hallucinating the FINAL answer belongs in E3, not E6.*
    """

    grading_rubric = """
    **Evaluation Rubric Part 2: Holistic Grade**
    
    * **1.0 (Correct):** All errors are 0.
    * **0.8 (Minor Format/Constraint Issue):** * The Logic is sound (it would work).
        * BUT contains **E3 (Simulated Execution)** or minor **E5 (Parameter Tuning)**.
        * Example: Plan is perfect, but the final step contains a made-up answer text.
    * **0.6 (Mostly Correct - Logical Flaw):** * The general strategy matches the Expert.
        * BUT contains **one** critical **E4 (Logical Defect)** or **E3 (Scope Violation)**.
        * Example: Good steps, but includes Frame 3000 when limit was 2000.
        * Example: Good steps, but Step 3 forgets to filter data before Step 4.
    * **0.4 (Partially Correct):** * Gets some steps right, but misses the main "Agentic Loop" or heavily hallucinates tools (**E6**).
    * **0.2 - 0.0:** Irrelevant or completely broken.
    """
    
    system_prompt = f"""
    You are an expert AI Agent Evaluator. Judge the `model_response` against the `user_query` and `expert_example`.

    {evaluation_philosophy}

    {evaluation_rubric}

    {grading_rubric}

    **Output Requirement:**
        * `is_correct`: (boolean) `true` if `error_list` is all zeros, `false` otherwise.
    * `grade`: (float) A score from `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, following the grading rubric.
    * `error_list`: (array) A 6-element array corresponding to errors [E1, E2, E3, E4, E5, E6]. The tested plan can be judged to have made multiple errors simultaneously.
    * `reasoning`: (string) A short concise explanation for the `error_list` and the assigned `grade`.


    Return ONLY a JSON object:
    {{
      "is_correct": boolean,
      "grade": float,
      "error_list": [E1, E2, E3, E4, E5, E6],
      "reasoning": "Concise justification. Cite specific step numbers and error codes."
    }}
    """
    return system_prompt

def create_framethinker_user_content(data):
    """
    Framethinker 逻辑：处理 <image> 标签分割文本，将图片插入文本中间。
    """
    # 1. 获取必要的文本信息
    generator_system_prompt = data.get('test_system_prompt', 'ERROR: test_system_prompt NOT FOUND IN DATA')
    gt_chain = json.dumps(data.get('ground_truth_tool_chain', 'N/A'), indent=2, ensure_ascii=False)
    
    # 2. 格式化模型回复
    model_resp_str = data['model_response']
    if isinstance(model_resp_str, dict):
        model_resp_str = json.dumps(model_resp_str, indent=2, ensure_ascii=False)
    else:
        try:
            parsed_resp = json.loads(model_resp_str)
            model_resp_str = json.dumps(parsed_resp, indent=2, ensure_ascii=False)
        except Exception:
            pass 

    # 3. 准备图片路径
    image_paths = []
    if 'background' in data and 'files' in data['background']:
        image_paths = [file_info['path'] for file_info in data['background']['files']]
        image_paths = [get_real_file_path('framethinker', img_path) for img_path in image_paths]

    # 4. 构建多模态 Content List
    content = []
    
    # --- Part A: 引导语和 User Query (包含图片) ---
    intro_text = "Here is the case to evaluate. Please provide your judgement based on the system prompt's rubric.\n\n--- 1. USER QUERY ---\n"
    content.append({"type": "text", "text": intro_text})
    
    query = data['query']
    query_parts = query.split("<image>")
    
    image_index = 0
    for i, part in enumerate(query_parts):
        # 添加文本部分
        if part:
            content.append({"type": "text", "text": part})
        
        # 如果不是最后一部分，说明这里有一个 <image> 标签被 split 切掉了，需要补上图片
        if i < len(query_parts) - 1:
            if image_index < len(image_paths):
                img_path = image_paths[image_index]
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Framethinker Image path does not exist: {img_path}")
                base64_image = encode_image_to_base64(img_path)
                
                if base64_image:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"
                        }
                    })
                else:
                    content.append({"type": "text", "text": "\n[Image Missing or Error]\n"})
                    
                image_index += 1
            else:
                content.append({"type": "text", "text": f"\n[Warning: Missing image file for <image> tag {i+1}]\n"})

    # --- Part B: 上下文 (System Prompt, GT, Model Response) ---
    context_text = f"""
    
    --- 2. GENERATOR'S SYSTEM PROMPT (Contains Tool Definitions) ---
    This is the *exact* prompt the model saw. Use it to understand the task's goals and constraints, and find the tool definitions within this text.
    {generator_system_prompt}

    --- 3. EXPERT EXAMPLE (FOR REFERENCE ONLY) ---
    * **Plan:** {data.get('ground_truth_plan', 'N/A')}
    * **Tool Chain:** {gt_chain}

    --- 4. MODEL'S GENERATED PLAN (TO BE JUDGED) ---
    {model_resp_str}
    """
    
    content.append({"type": "text", "text": context_text})
    
    return content

def create_gaia_system_prompt():    
    evaluation_philosophy = """
    **Your Evaluation Philosophy: The "Generous Execution" Standard**
    You are an expert AI Agent Evaluator. Your goal is to determine if the plan is **Functionally Executable**, not if it is syntactically perfect. You must apply a "Generous Interpreter" mindset to bridge the gap between natural language planning and strict code execution.

    **1. PRINCIPLE OF TOOL INTELLIGENCE (Capability Assumption):**
       - **Assume Smart Tools:** When judging tool usage, assume the underlying tools are robust and capable of handling ambiguity. If a tool's name or description suggests it *could* reasonably perform a requested action, **assume it works**.
       - **Intent > Precision:** Do not penalize the agent for using imprecise parameter names or formatting if the *semantic intent* is clear and the logic is sound. Only mark a failure if the tool is being used for a fundamentally wrong category of action (e.g., using a calculation tool to perform a web search).

    **2. PRINCIPLE OF LOGICAL CONTINUITY (Data Flow):**
       - **Implicit Context:** Planning often involves implicit steps (e.g., "finding" implies "selecting"). If Step A generates data and Step B needs it, assume the data flows correctly even without explicit variable binding.
       - **Redundancy is Valid:** Strategies that involve re-checking, re-fetching, or multiple verification steps are **valid and robust**. Do not penalize them as "inefficient" or "incorrect".

    **3. PRINCIPLE OF ZERO KNOWLEDGE (The Hard Line):**
       - **The Oracle Barrier:** This is the only rule where you must be strict. The plan must reflect the state of knowledge *before* execution.
       - **Hallucination vs. Placeholder:** * **Fail (E6):** Hardcoding specific external facts, numbers, or filenames that are not in the user context.
         * **Pass:** Using descriptive placeholders (e.g., "The value found in the previous step") or logical estimates explicitly marked as assumptions.
    """

    evaluation_rubric = """
    **Evaluation Rubric: Universal Error Categories**

    **1. E1_GOAL_MISALIGNMENT:** The plan solves a different problem than what was requested.
    
    **2. E2_INCOMPLETE_COVERAGE:** A significant, distinct part of the user's request is completely ignored. *Note: Minor omissions of non-critical details should be graded as 0.8, not E2.*

    **3. E3_CONSTRAINT_VIOLATION:** Explicit negative constraints in the prompt are violated.

    **4. E4_FATAL_LOGIC_BREAK:** - **Definition:** The chain of causality is objectively broken. Step B requires input that Step A cannot possibly provide (e.g., temporal paradoxes or fundamental data type mismatches).
       - **Exemption:** Optimistic assumptions about search results or tool capabilities are **NOT** logic breaks.

    **5. E5_TOOL_USE_ERROR:**
       - **Definition:** Calling a tool to perform an action that is completely outside its defined capabilities domain.
       - **Exemption:** Parameter syntax errors, invented parameter names that convey correct intent, or ambiguous formatting are **Pass (1.0)**.

    **6. E6_FACTUAL_HALLUCINATION:**
       - **Definition:** The plan treats unknown future results as known current facts.
       - **Test:** Does the plan contain specific information (values, names) that could only be known *after* executing the tools?
    """

    grading_rubric = """
    **Grading Scale:**
    - **1.0 (Correct):** Plan is executable and logical. **Redundancy is OK.**
    - **0.8 (Minor Flaw):** Viable, but maybe uses a tool slightly weirdly (but still workable).
    - **0.6 (Mostly Correct):** Solves the main goal but misses a specific constraint (e.g., "Exclude references") or has a minor logic gap.
    - **0.4 (Broken):** Fatal logic break (E4) or Major Hallucination (E6).
    - **0.0 - 0.2:** Failure.
    """
    
    system_prompt = f"""
    You are an expert Evaluator of AI Agent Planning.
    
    {evaluation_philosophy}

    {evaluation_rubric}

    {grading_rubric}

    **Output Format Requirement:**
    You must return your response *only* as a JSON object, without any explanatory text or markdown.
    The JSON object must have the following **four** keys: `is_correct`, `grade`, `error_list`, and `reasoning`.

    * `is_correct`: (boolean) `true` if `error_list` is all zeros, `false` otherwise.
    * `grade`: (float) A score from `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, following the grading rubric.
    * `error_list`: (array) A 6-element array corresponding to errors [E1, E2, E3, E4, E5, E6]. The tested plan can be judged to have made multiple errors simultaneously.
    * `reasoning`: (string) A short concise explanation for the `error_list` and the assigned `grade`.

    **Example Output:**
    {{
      "is_correct": boolean,
      "grade": float,
      "error_list": [E1, E2, E3, E4, E5, E6],
      "reasoning": "Your reason for making this judgment."
    }}
    """
    return system_prompt

def create_gaia_user_content(data):
    """
    Gaia 逻辑：文本在前，支持读取 TXT 文件内容作为文本块，图片统一放在最后。
    """
    
    # 1. 获取 test_system_prompt
    generator_system_prompt = data.get('test_system_prompt', 'ERROR: test_system_prompt NOT FOUND IN DATA')
    
    # 2. 格式化 GT
    gt_chain = json.dumps(data.get('ground_truth_tool_chain', 'N/A'), indent=2, ensure_ascii=False)
    
    # 3. 格式化模型回复
    model_resp_str = data['model_response']
    if isinstance(model_resp_str, dict):
        model_resp_str = json.dumps(model_resp_str, indent=2, ensure_ascii=False)
    else:
        try:
            # 尝试解析，使其格式更美观，如果失败也无妨
            parsed_resp = json.loads(model_resp_str)
            model_resp_str = json.dumps(parsed_resp, indent=2, ensure_ascii=False)
        except Exception:
            pass 

    # --- 构造文本部分 ---
    text_content = f"""
    Here is the case to evaluate. Please provide your judgement based on the system prompt's rubric.

    --- 1. USER QUERY ---
    {data['query']}

    --- 2. GENERATOR'S SYSTEM PROMPT (Contains Tool Definitions) ---
    This is the *exact* prompt the model saw. Use it to understand the task's goals and constraints, and find the tool definitions within this text.
    {generator_system_prompt}

    --- 3. EXPERT EXAMPLE (FOR REFERENCE ONLY) ---
    * **Plan:** {data.get('ground_truth_plan', 'N/A')}
    * **Tool Chain:** {gt_chain}

    --- 4. MODEL'S GENERATED PLAN (TO BE JUDGED) ---
    {model_resp_str}
    """

    # --- [修改] 构建 Multimodal 内容列表 ---
    content_list = []

    # 1. 首先加入核心提示词文本
    content_list.append({"type": "text", "text": text_content})

    # 2. 从 background 中提取文件并加入
    files_data = data['background']['files']
    if files_data:
        files_list = data['background'].get('files', [])
        
        for file_info in files_list:
            file_path = file_info['path']
            file_path = get_real_file_path('gaia', file_path)
            low_case_path = file_path.lower()
            
            # 严格检查文件是否存在
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"CRITICAL: File required for Judge not found: {file_path}")

            # --- [新增] 处理 TXT 文件 ---
            if low_case_path.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    # 将 TXT 内容作为一个新的 text 块附加
                    content_list.append({
                        "type": "text",
                        "text": f"\n--- Start of provided file content: {file_path} ---\n{file_content}\n--- End of provided file content ---\n"
                    })
                except Exception as e:
                    print(f"CRITICAL: Failed to read text file {file_path} for Judge. Error: {e}")
                    raise e
            
            # --- [原有] 处理图片文件 ---
            elif low_case_path.endswith(('.jpg', '.png', '.jpeg')):
                try:
                    base64_image = encode_image_to_base64(file_path)
                    
                    # Gaia 逻辑严格检查：如果图片读取/编码失败（返回None），则手动抛出异常
                    if base64_image is None:
                         raise ValueError("Encoded image is None")

                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto" 
                        }
                    })
                except Exception as e:
                    # [关键] 根据要求：如果图片读取/编码失败，则直接抛出异常
                    print(f"CRITICAL: Failed to encode image {file_path} for Judge. Error: {e}")
                    raise e 

    return content_list


def create_gta_system_prompt():
    evaluation_philosophy = """
    **Your Evaluation Philosophy (Crucial for Planning Agents):**
    You are an expert Judge evaluating an AI Agent's ability to create a logical **PLAN**.
    
    **1. THE "NO PEEKING" RULE (CRITICAL - MOST COMMON FAILURE):**
       - The Agent CANNOT know specific values from an image (e.g., a price "$19.99", a name "Magna", a count "4") BEFORE it calls a tool (like OCR or Count) to extract them.
       - **Immediate Fail (0.4 - 0.6):** If the plan references a specific string/number from the image in a parameter or reason *before* the extraction step outputs it.
       - **Correct:** Reference the *concept* (e.g., `item_price`, `extracted_text`).
       - **Incorrect:** Hardcoding the value (e.g., `479.99`, `Regency Cafe`). **This is a Fatal Error (E3/E6).**

    **2. BLUEPRINT vs. SYNTAX:**
       - **Accept:** Pseudo-code logic in parameters (e.g., `Calculator(expression="price * 1.15")`). Even if variables aren't declared, if the *data flow* is clear, it is correct.
       - **Accept:** Conditional logic described in text (e.g., "Apply discount if count > 4").
       - **Reject:** Only reject if the logic is functionally impossible (e.g., trying to calculate a total before finding the price).

    **3. GENERATOR CONSTRAINTS:**
       - You must enforce the constraints defined in the "GENERATOR'S SYSTEM PROMPT".
       - Specifically, look for the **"STRICT Data Access Rule"**. If the agent violates this by incorporating visual details into the plan before extracting them, mark it as **E3 (Constraint Violation)**.

    **4. DEPENDENCY VERIFICATION:**
       - Verify that Step N only uses data available from Steps 1 to N-1 or the user query.
       - If Step 3 uses "The breed identified in Step 2", check if Step 2 actually returns a breed. If Step 2 only returns a bounding box, this is a **Logical Defect (E4)**.
    """

    evaluation_rubric = """
    **Evaluation Rubric: Error Catalog**

    **1. E1_GOAL_UNDERSTANDING:** Fails to address the core user query.
    
    **2. E2_TASK_COMPLETENESS:** Misses a distinct sub-task (e.g., user asked for "Tax AND Conversion", agent only did Tax).

    **3. E3_CONSTRAINT_VIOLATION (Strict Data Access):** - **CRITICAL:** Incorporating specific visual information (prices, names, counts) into the plan *before* using a tool to extract it. 
       - Example: Calling `GoogleSearch(query="Regency Cafe")` as the first step when the user only gave a photo of a building. The agent should have used OCR first.

    **4. E4_LOGICAL_DEFECT (Broken Dependency):**
       - **Definition:** The plan is functionally broken because a step relies on data that has not been generated yet.
       - **Example:** Calculating `total` before `quantity` is known.
       - **Example:** Using a variable that previous tools do not output (e.g., using `weight` when the previous tool only returned `size`).

    **5. E5_TOOL_USE_ERROR (Functional Misuse):**
       - Using a tool for the wrong purpose (e.g., using `Calculator` to parse text).
       - Note: Minor variable naming issues are NOT E5.

    **6. E6_HALLUCINATION_ERROR (Fact Fabrication):**
       - **Definition:** Hardcoding a result that should be dynamic.
       - **Example:** `Calculator(expression="4 * 15.5")` where "15.5" is a value from the image that hasn't been read yet.
       - **Constraint:** If the value exists in the User Query, it is NOT hallucination. If it only exists in the Image, it IS hallucination.
    """

    grading_rubric = """
    **Grading Scale:**
    - **1.0 (Correct):** Plan is logically sound. Follows dependency rules. No "peeking" at image data.
    - **0.8 (Minor Flaw):** Plan is logical and solvable, but has minor inefficiency or slight syntax ambiguity. **NO E3 or E6 errors allowed here.**
    - **0.6 (Mostly Correct):** Solves the main task but fails a sub-part OR has a genuine logical gap (E4) or slight data leakage (minor E3).
    - **0.4 (Broken):** Fatal logic break (E4), Major Hallucination (E6), or Clear Violation of Strict Data Access (E3).
    - **0.0 - 0.2:** Complete Failure.
    """
    
    system_prompt = f"""
    You are an expert Evaluator of AI Agent Planning.

    {evaluation_philosophy}

    {evaluation_rubric}

    {grading_rubric}

    **Output Format Requirement:**
    You must return your response *only* as a JSON object, without any explanatory text or markdown.
    The JSON object must have the following **four** keys: `is_correct`, `grade`, `error_list`, and `reasoning`.

    * `is_correct`: (boolean) `true` if `error_list` is all zeros, `false` otherwise.
    * `grade`: (float) A score from `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, following the grading rubric.
    * `error_list`: (array) A 6-element array corresponding to errors [E1, E2, E3, E4, E5, E6]. The tested plan can be judged to have made multiple errors simultaneously.
    * `reasoning`: (string) A short concise explanation for the `error_list` and the assigned `grade`.

    **Example Output:**
    {{
      "is_correct": boolean,
      "grade": float,
      "error_list": [E1, E2, E3, E4, E5, E6],
      "reasoning": "Your reason for making this judgment."
    }}
    """
    return system_prompt

def create_gta_user_content(data):
    # 1. (新) 获取 test_system_prompt
    # (我们将在 process_judgement 中检查它是否存在)
    generator_system_prompt = data.get('test_system_prompt', 'ERROR: test_system_prompt NOT FOUND IN DATA')
    
    # 2. 格式化 GT
    gt_chain = json.dumps(data.get('ground_truth_tool_chain', 'N/A'), indent=2, ensure_ascii=False)
    
    # 3. 格式化模型回复
    model_resp_str = data['model_response']
    if isinstance(model_resp_str, dict):
        model_resp_str = json.dumps(model_resp_str, indent=2, ensure_ascii=False)
    else:
        try:
            # 尝试解析，使其格式更美观，如果失败也无妨
            parsed_resp = json.loads(model_resp_str)
            model_resp_str = json.dumps(parsed_resp, indent=2, ensure_ascii=False)
        except Exception:
            pass # 如果失败，就用原始字符串

    # --- 构造文本部分 ---
    text_content = f"""
    Here is the case to evaluate. Please provide your judgement based on the system prompt's rubric.

    --- 1. USER QUERY ---
    {data['query']}

    --- 2. GENERATOR'S SYSTEM PROMPT (Contains Tool Definitions) ---
    This is the *exact* prompt the model saw. Use it to understand the task's goals and constraints, and find the tool definitions within this text.
    {generator_system_prompt}

    --- 3. EXPERT EXAMPLE (FOR REFERENCE ONLY) ---
    * **Plan:** {data.get('ground_truth_plan', 'N/A')}
    * **Tool Chain:** {gt_chain}

    --- 4. MODEL'S GENERATED PLAN (TO BE JUDGED) ---
    {model_resp_str}
    """

    # --- [修改] 构建 Multimodal 内容列表 ---
    content_list = []

    # 1. 首先加入文本
    content_list.append({"type": "text", "text": text_content})

    # 2. 从 background 中提取图片并加入
    # 注意：输入数据是 test 脚本的输出结果，其中包含 'background'
    files_data = data.get('background')['files']
    if files_data:
        image_paths = [file_info['path'] for file_info in data['background']['files']]
        image_paths = [get_real_file_path('gta', img_path) for img_path in image_paths]
        
        for img_path in image_paths:
            try:
                base64_image = encode_image_to_base64(img_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto" 
                    }
                })
            except Exception as e:
                print(f"CRITICAL: Failed to encode image {img_path} for Judge. Error: {e}")
                raise e 

    return content_list

def create_skywork_system_prompt():
    evaluation_philosophy = """
    **Your Evaluation Philosophy (Strict Logic & Protocol Adherence):**
    You are evaluating an AI Agent's plan in a **non-interactive, batch simulation context**. Your goal is to separate "Perfect Execution" from "Merely Functional" or "Non-Compliant" execution.

    **1. THE "AUTOPILOT" RULE (The ONLY Exemption):**
       - **Exempt:** The Agent does NOT need to stop and wait for user confirmation (e.g., after `stage_todo_card`). Proceeding immediately is CORRECT behavior for this test.
       - **Non-Exempt:** ALL other constraints in the System Prompt MUST be followed rigidly.

    **2. "GATHER BEFORE WRITE" (FATAL LOGIC CHECK):**
       - **Rule:** An agent CANNOT write a specific, factual report without first calling a research tool (e.g., `doc_task_agent`).
       - **Verdict:** Jumping straight to `write_agent` without research is a **Fatal Logical Defect (E4)**. Grade must be **<= 0.4**.

    **3. STRICT PROTOCOL ENFORCEMENT (E3 - Constraint Violation):**
       - **The Rule:** If the System Prompt dictates a specific tool for an action (e.g., "Update TODO.md with `write_file` tool"), the Agent MUST use that exact tool.
       - **The Violation:** - Skipping the update entirely? -> **E3 Error**.
         - Updating it via a different method (e.g., using `doc_task_agent` to run a shell command, or using a non-standard tool)? -> **E3 Error**.
       - **Verdict:** Even if the file theoretically gets updated, using the wrong mechanism is a compliance failure. Max Grade **0.8**.

    **4. TASK COMPLETENESS (E2):**
       - If the plan creates a TODO list but stops without executing ANY research/writing steps, it is **Incomplete (E2)**. Grade **0.4**.
    """

    evaluation_rubric = """
    **Evaluation Rubric: Error Catalog**
    Set 1 (Error Present) or 0 (Error Absent) for EACH category.

    **1. E1_GOAL_UNDERSTANDING:** Fails to address the core user query.

    **2. E2_TASK_COMPLETENESS:** - Misses a key deliverable or stops at the planning phase.

    **3. E3_CONSTRAINT_VIOLATION (Strict SOPs):**
       - **Flag = 1 IF:**
         - The agent failed to update intermediate files (e.g., `TODO.md`) as required.
         - **The agent used the WRONG tool/method** to perform a mandatory action (e.g., prompted to use `write_file` but used something else).
         - The agent used a tool prohibited by the stage.
       - **EXCEPTION:** Do NOT flag for "not waiting for user".

    **4. E4_LOGICAL_DEFECT (Dependency & Flow):**
       - **Critical:** Writing without Researching.
       - **Critical:** Using variables that don't exist.

    **5. E5_TOOL_USE_ERROR:**
       - Invalid parameter structures or malformed arguments.

    **6. E6_HALLUCINATION_ERROR:**
       - Inventing tools or fabricating specific data results.
    """
    
    grading_rubric = """
    **Evaluation Rubric Part 2: Holistic Grade**

    **1. Grade 1.0 (Perfect / Near Perfect):**
       - The plan is logically sound, executes the research, and produces the final output.
       - Ignores "wait for user" correctly.

    **2. Grade 0.8 (Functional but Minor Flaws):**
       - The plan solves the problem, but misses **administrative** steps (e.g., forgot to update `TODO.md` status).
       - The logic is valid, but the process hygiene is slightly off. **This is a very common grade.**

    **3. Grade 0.6 (Major Flaw):**
       - Significant tool misuse (e.g., using the wrong agent for a stage).
       - Or major inefficiency (e.g., researching the same thing 3 times).

    **4. Grade 0.4 (Broken/Incomplete):**
       - **Fatal:** Skips the "Research" phase entirely and tries to write immediately (Logic Break).
       - **Fatal:** Stops after the "Planning" phase and never executes the work.

    **5. Grade 0.0 - 0.2:** - Complete failure to understand the task or hallucinated tools.
    """
    
    system_prompt = f"""
    You are an expert AI Agent Auditor. 
    Your goal is to ensure the Agent is **Compliant** with its System Prompt instructions and **Logically Sound**.

    {evaluation_philosophy}

    {evaluation_rubric}

    {grading_rubric}

    **Output Format Requirement:**
    You must return your response *only* as a JSON object, without any explanatory text or markdown.
    The JSON object must have the following **four** keys: `is_correct`, `grade`, `error_list`, and `reasoning`.

    * `is_correct`: (boolean) `true` if `error_list` is all zeros, `false` otherwise.
    * `grade`: (float) A score from `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, following the grading rubric.
    * `error_list`: (array) A 6-element array corresponding to errors [E1, E2, E3, E4, E5, E6]. The tested plan can be judged to have made multiple errors simultaneously.
    * `reasoning`: (string) A short concise explanation for the `error_list` and the assigned `grade`.

    **Example Output:**
    {{
      "is_correct": boolean,
      "grade": float,
      "error_list": [E1, E2, E3, E4, E5, E6],
      "reasoning": "Your reason for making this judgment."
    }}
    """
    return system_prompt

def create_skywork_user_content(data):
    # 1. (新增) 获取被测模型所使用的 task-specific system prompt
    task_system_prompt = data.get('test_system_prompt', 'N/A (WARNING: system_prompt not found in new data format)')

    # 3. 格式化 GT
    gt_chain = json.dumps(data.get('ground_truth_tool_chain', 'N/A'), indent=2, ensure_ascii=False)
    
    # 4. 格式化模型回复
    model_resp_str = data['model_response']
    if isinstance(model_resp_str, dict):
        model_resp_str = json.dumps(model_resp_str, indent=2, ensure_ascii=False)
    else:
        try:
            parsed_resp = json.loads(model_resp_str)
            model_resp_str = json.dumps(parsed_resp, indent=2, ensure_ascii=False)
        except Exception:
            pass # 如果失败，就用原始字符串

    content = f"""
    Here is the case to evaluate. Please provide your judgement based on the system prompt's rubric.
        --- 1. USER QUERY ---
    {data['query']}

    --- 2. TASK-SPECIFIC SYSTEM PROMPT (for context) ---
    This was the system_prompt given to the model being tested. Use it to understand the task's goals and constraints, and find the tool definitions within this text.
    {task_system_prompt}

    --- 3. EXPERT EXAMPLE (FOR REFERENCE ONLY) ---
    * **Plan:** {data.get('ground_truth_plan', 'N/A')}
    * **Tool Chain:** {gt_chain}

    --- 4. MODEL'S GENERATED PLAN (TO BE JUDGED) ---
    {model_resp_str}
    """
    return content

def create_tool_bench_system_prompt():
    evaluation_philosophy = """
    **Your Evaluation Philosophy (Crucial):**
    Your primary goal is to act as an expert logical verifier. You must determine if the `model_response` (containing a 'tool_chain') is **logically capable of *solving* the `user_query`**.

    1.  **Check for "Solvability," not "Similarity":** A plan that *looks* internally consistent but *cannot* lead to the correct answer (e.g., due to a flawed sequence or tool omission) is a **failed plan**.
    2.  **Use the Expert Example as a *Reference for Logic*: ** The `expert_example` shows *one valid way* to solve the problem. The model does *not* have to match it. **However,** if the model's `tool_chain` is *fundamentally different* (e.g., it *omits* a tool that the expert *requires*), you **must** critically analyze *why*. Is the model's new path *also* logically viable? Or did it *miss the logical point*?
    3. **Detect MULTIPLE Errors:** A plan often fails in multiple ways. **Do NOT stop after finding the first error.** You must evaluate all 6 categories independently. A plan can simultaneously have E2 (Incomplete), E4 (Logical Break), and E6 (Hallucination).
    4.  **Find Tools in the Generator's Prompt:** You will be given the `GENERATOR'S SYSTEM PROMPT`. This prompt contains the **"Available Tools"** section. You **must** use that section as the single source of truth for tool names, functions, and parameters.
    5.  **Evaluate the *Sequence*:** Your focus is on the *linear sequence* of the `tool_chain`. Check for correct data dependencies (e.g., data from Step 1 is used correctly in Step 2) and logical omissions.
    """

    evaluation_rubric = """
    **Evaluation Rubric Part 1: 6-Category Error Catalog**
    Evaluate the `model_response` for the following 6 error types. **Set 1 (Error Present) or 0 (Error Absent) for EACH category independently.**

    --- Error Catalog ---

    **1. E1_GOAL_UNDERSTANDING:**
        - **Definition:** Fundamentally misunderstands the `user_query`'s core intent.
        - **Example:** Query asks to "summarize the Q3 financial report", but the plan tries to "delete all customer reviews".

    **2. E2_TASK_COMPLETENESS:**
        - **Definition:** Fails to plan for all required sub-tasks of a multi-part query.
        - **Example:** Query asks to "find the document, summarize it, and email the summary", but the `tool_chain` only plans to "find" and "summarize", completely omitting the "email" step.

    **3. E3_CONSTRAINT_VIOLATION:**
        - **Definition:** The plan violates an explicit *constraint* from the `user_query` or the task-specific `SYSTEM PROMPT`.
        - **Example:** Query says "create a 100-word summary", but the plan calls `summarize_text` with the `parameter_description` set to "length: 500 words".

    **4. E4_LOGICAL_DEFECT:**
        - **Definition:** The logical reasoning in the plan or tool chain doesn't match, the execution steps lack key premises, assumptions, and conditions, or there are circular arguments, making the query *unsolvable*.
        - **Example 1:** Tries to *use* a 'document_id' in Step 2 (e.g., in `get_document_content`) before the step that *finds* it in Step 3 (e.g., `search_documents`).
        - **Example 2:** The query requires finding and then summarizing a document, but the plan *omits* the `get_document_content` tool, trying to pass a 'document_id' directly to `summarize_text` (which expects 'text').

    **5. E5_TOOL_USE_ERROR:**
        - **Definition:** The plan misunderstands a real tool's *function* or its required *data type*.
        - **Example:** `parameter_description` shows it is passing a 'list of document objects' (from `search_documents`) to the `summarize_text` tool, which requires a single 'text' string.
    
    **6. E6_HALLUCINATION_ERROR:**
        - **Definition:** The plan calls a non-existent tool (a tool not listed in the `GENERATOR'S SYSTEM PROMPT`'s "Available Tools" list) or outputs factual errors that are clearly contrary to common sense. Or, the plan uses results that are not yet available at the current step.
        - **Example:** Calls a fake tool `CalculateTotalCost` or `FindCheapestWineFromMenu` when only `OCR` and `CountGivenObject` are available.
        - **Example:** `parameter_description` contains a specific *answer* (e.g., `"cheapest_wine_price": 25.00`) that is impossible to know before executing `OCR`.

    """
    grading_rubric = """
    **Evaluation Rubric Part 2: Holistic Grade (0.0-1.0)**
    After filling the `error_list`, you must assign a `grade`.

    **1. The "Correct" Case (Grade 1.0):**
    * If the `error_list` is all zeros (`[0, 0, 0, 0, 0, 0]`), then `is_correct` **must** be `true` and the `grade` **must** be `1.0`.

    **2. The "Incorrect" Case (Grade 0.0 - 0.8):**
    * If the `error_list` contains *any* '1', then `is_correct` **must** be `false`.
    * You must then assign a partial `grade` from the set `[0.0, 0.2, 0.4, 0.6, 0.8]`.
    * To assign this grade, you **must compare the model's `tool_chain` to the `expert_example`'s `tool_chain`** to judge partial correctness.

    **Partial Grade Definitions:**
    * **`grade: 0.8` (Very Good, Minor Flaw):** The plan is *almost* perfect. It follows the expert logic very closely, but has a minor, non-critical error. The core logic is 90% correct.
    * **`grade: 0.6` (Mostly Correct):** The plan captures the *main* logical flow (e.g., 2 out of 3 key steps are correct) but it *misses* or *flaws* one key component (`E4` or `E5` on a critical step). The plan is "on the right track" but fails.
    * **`grade: 0.4` (Partially Correct):** The plan identifies *some* correct steps (e.g., the first step is right, or it has 2 of 4 steps right), but the overall sequence is wrong or fundamentally broken. It shows *some* understanding but is largely incorrect.
    * **`grade: 0.2` (Mostly Incorrect):** The plan is deeply flawed. It might get *one* simple step right (like `find_object`) but completely fails to address the *core* logic of the query (`E1`, `E2`, or a total `E4` failure).
    * **`grade: 0.0` (Completely Incorrect):** The plan has zero logical merit. It is complete hallucination (`E6`), completely misunderstands the goal (`E1`), or is an empty/useless response.
    """
    
    system_prompt = f"""
    You are an expert, impartial, and meticulous AI Agent Evaluator (a "Judge").
    Your task is to judge the quality of an AI-generated plan (`model_response`) for a given `user_query`, based on the provided rubric. The `model_response` should contain a JSON with a "plan" and a "tool_chain" list.

    {evaluation_philosophy}

    {evaluation_rubric}

    {grading_rubric}

    **Output Format Requirement:**
    You must return your response *only* as a JSON object, without any explanatory text or markdown.
    The JSON object must have the following **four** keys: `is_correct`, `grade`, `error_list`, and `reasoning`.

    * `is_correct`: (boolean) `true` if `error_list` is all zeros, `false` otherwise.
    * `grade`: (float) A score from `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, following the grading rubric.
    * `error_list`: (array) A 6-element array corresponding to errors [E1, E2, E3, E4, E5, E6].
    * `reasoning`: (string) A concise explanation for the `error_list` and the assigned `grade`. **You must justify the grade.**

    **Example Output (if incorrect):**
    {{
      "is_correct": false,
      "grade": 0.4,
      "error_list": [0, 0, 0, 1, 0, 1],
      "reasoning": "The plan is Incorrect. It fails on E4_LOGICAL_DEFECT because the tool chain omits the 'choose_frames_between' tool, which is logically necessary. This plan also commits an E6 error, which contains factual errors. Grade 0.4 assigned because while the first step was correct, the core logic of 'zooming in' was missed entirely."
    }}

    **Example Output (if correct):**
    {{
      "is_correct": true,
      "grade": 1.0,
      "error_list": [0, 0, 0, 0, 0, 0],
      "reasoning": "The plan is Correct (Grade 1.0). It correctly understood the query, and the tool chain presents a logical and efficient sequence to solve the task."
    }}
    """
    return system_prompt

def create_tool_bench_user_content(data):
    generator_system_prompt = data.get("test_system_prompt", "ERROR: system_prompt NOT FOUND IN DATA")
    
    # 2. 格式化 GT
    gt_chain = json.dumps(data.get('ground_truth_tool_chain', 'N/A'), indent=2, ensure_ascii=False)
    
    # 3. 格式化模型回复
    model_resp_str = data['model_response']
    if isinstance(model_resp_str, dict):
        model_resp_str = json.dumps(model_resp_str, indent=2, ensure_ascii=False)
    else:
        try:
            # 尝试解析，使其格式更美观，如果失败也无妨
            parsed_resp = json.loads(model_resp_str)
            model_resp_str = json.dumps(parsed_resp, indent=2, ensure_ascii=False)
        except Exception:
            pass # 如果失败，就用原始字符串

    content = f"""
    Here is the case to evaluate. Please provide your judgement based on the system prompt's rubric.
    --- 1. USER QUERY ---
    {data['query']}

    --- 2. GENERATOR'S SYSTEM PROMPT (Contains Tool Definitions) ---
    This is the *exact* prompt the model saw. Use it to understand the task's goals and constraints, and find the tool definitions within this text.
    {generator_system_prompt}

    --- 3. EXPERT EXAMPLE (FOR REFERENCE ONLY) ---
    * **Plan:** {data.get('ground_truth_plan', 'N/A')}
    * **Tool Chain:** {gt_chain}

    --- 4. MODEL'S GENERATED PLAN (TO BE JUDGED) ---
    {model_resp_str}

    """
    return content

# ==========================================
# 3. 评估核心逻辑 (合并版)
# ==========================================

def get_judge_generation(system_prompt, user_content, client, judge_model_name):
    """
    通用 API 调用函数
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # 非流式输出
    response = client.chat.completions.create(
        model=judge_model_name,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=30240,
        temperature=0.0
    )
    
    model_response_string = response.choices[0].message.content
    return model_response_string

def process_judgement(line, log_path, lock, client, judge_model_name):
    """
    核心处理函数：
    1. 解析 JSON。
    2. 检查 source_dataset，判断使用 Framethinker 还是 Gaia 的逻辑。
    3. 调用 API 进行 Judge。
    4. 写入结果。
    """
    data = None
    index = "UNKNOWN"
    judge_response_string = "" 
    cleaned_response_string = "" 
    
    try:
        data = json.loads(line.strip())
        index = data.get('index', 'UNKNOWN')
        # 获取数据集标识
        source_dataset = data.get('source_dataset', '').lower()

        if 'test_system_prompt' not in data:
            raise ValueError(f"CRITICAL: 'test_system_prompt' key not found in input line for index {index}.")

        # --- 分发逻辑 (Dispatcher) ---
        system_prompt = ""
        user_content = []

        
        if 'agentnetbench' in source_dataset:
            system_prompt = create_agentnetbench_system_prompt()
            user_content = create_agentnetbench_user_content(data)
        elif 'framethinker' in source_dataset:
            system_prompt = create_framethinker_system_prompt()
            user_content = create_framethinker_user_content(data)
        elif 'gaia' in source_dataset:
            system_prompt = create_gaia_system_prompt()
            user_content = create_gaia_user_content(data)
        elif 'gta' in source_dataset:
            system_prompt = create_gta_system_prompt()
            user_content = create_gta_user_content(data)
        elif 'skywork' in source_dataset:
            system_prompt = create_skywork_system_prompt()
            user_content = create_skywork_user_content(data)
        elif 'tool_bench' in source_dataset:
            system_prompt = create_tool_bench_system_prompt()
            user_content = create_tool_bench_user_content(data)
        
        else:
            # 如果无法识别，记录错误并跳过
            raise ValueError(f"Unknown source_dataset: '{source_dataset}'. Expected 'framethinker' or 'gaia/gta'.")

        # --- 调用 LLM ---
        judge_response_string = get_judge_generation(
            system_prompt, 
            user_content,
            client,
            judge_model_name
        )
        
        try:
            # 清理 Judge 的回复
            cleaned_response_string = judge_response_string.strip()
            
            if cleaned_response_string.startswith("```json"):
                cleaned_response_string = cleaned_response_string[len("```json"):].strip()
            elif cleaned_response_string.startswith("```"):
                cleaned_response_string = cleaned_response_string[len("```"):].strip()
            
            if cleaned_response_string.endswith("```"):
                cleaned_response_string = cleaned_response_string[:-len("```")].strip()
            
            judge_json = json.loads(cleaned_response_string) 
            
        except json.JSONDecodeError as e:
            error_message = (
                f"Judge response was not valid JSON. Error: {e}. "
                f"Raw response: {judge_response_string} | "
                f"Cleaned response: {cleaned_response_string}"
            )
            print(f"ERROR on index {index} ({source_dataset}): {error_message}. Skipping.")
            log_error(index, error_message, log_path, lock) 
            return None

        # 将 'judgement' 附加到原始数据中
        data['judgement'] = judge_json
        return data
        
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        print(f"ERROR on index {index}: {error_message}. Skipping.")
        log_error(index, error_message, log_path, lock) 
        return None

# ==========================================
# 4. 文件与并发管理 (Shared)
# ==========================================

def create_log_dir(log_dir, log_path):
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logging errors to: {log_path}")
    except Exception as e:
        print(f"CRITICAL: Could not create log directory {log_dir}. Error: {e}")
        raise
def load_processed_indices(output_path):
    """
    读取已存在的结果文件
    获取已完成 Judgement 的 (index, source_dataset) 集合以便跳过
    """
    processed = set()
    if not os.path.exists(output_path):
        return processed
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 必须包含 index 和 judgement 字段才视为有效处理过的记录
                    if 'index' in data and 'judgement' in data:
                        idx = data.get('index')
                        src = data.get('source_dataset')
                        
                        # 只要 index 存在，就加入集合（source_dataset 允许为 None，但通常应该有值）
                        if idx is not None:
                            processed.add((idx, src))
                            
                except json.JSONDecodeError:
                    continue
        if processed:
            print(f"Found {len(processed)} already judged items. They will be skipped.")
    except Exception as e:
        print(f"Warning: Could not read judged items from {output_path}. Error: {e}")
    return processed

def log_error(index, error_message, log_path, lock):
    with lock:
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] Index: {index} | Error: {error_message}\n")
        except Exception as e:
            print(f"CRITICAL: Failed to write to log file {log_path}. Error: {e}")
def run_judgement(input_path, output_path, log_dir, log_path, lock, client, judge_model_name, max_workers, data_num=-1):
    create_log_dir(log_dir, log_path)
    
    # 1. 加载已处理的 (index, source_dataset) 集合
    processed_keys = load_processed_indices(output_path)
    
    print(f"Starting Judgement...\nJudge Model: {judge_model_name}\nInput: {input_path}\nOutput: {output_path}")
    
    lines_to_process = []
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            line_iterator = infile
            if data_num > 0:
                line_iterator = itertools.islice(infile, data_num)
            for line in line_iterator:
                try:
                    data = json.loads(line)
                    
                    # 2. 获取当前行的标识
                    current_idx = data.get('index')
                    current_src = data.get('source_dataset')
                    current_key = (current_idx, current_src)
                    
                    # 3. 检查联合键是否已存在
                    if current_key not in processed_keys:
                        lines_to_process.append(line)
                        
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"CRITICAL: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"CRITICAL: Failed to read input file. Error: {e}")
        return

    total_to_process = len(lines_to_process)
    if total_to_process == 0:
        print("No new items to judge.")
        return
        
    print(f"Total items to judge: {total_to_process} (Skipped {len(processed_keys)} existing items)")
    
    with open(output_path, 'a', encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_line = {
                executor.submit(
                    process_judgement, 
                    line, 
                    log_path, 
                    lock, 
                    client, 
                    judge_model_name
                ): line 
                for line in lines_to_process
            }
            
            processed_count = 0
            for future in as_completed(future_to_line):
                result = future.result()
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()
                
                processed_count += 1
                if total_to_process > 0:
                    print(f"Progress: {processed_count}/{total_to_process} ({(processed_count/total_to_process)*100:.1f}%)")

    print(f"Judgement complete. Results appended to {output_path}")

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    
    MODEL_CONFIGS = {
        "gemini": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "openai_gpt": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "claude": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        },
        "default": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
        }
    }
    
    # 设定 Judge 模型
    JUDGE_MODEL_NAME = "gemini-3-pro-preview"
    # JUDGE_MODEL_NAME = "claude-sonnet-4-5-20250929"

    if "gemini" in JUDGE_MODEL_NAME:
            config_key = "gemini"
    elif "gpt" in JUDGE_MODEL_NAME:
        config_key = "openai_gpt"
    elif "claude" in JUDGE_MODEL_NAME:
        config_key = "claude"
    
    API_KEY = MODEL_CONFIGS[config_key]['api_key']
    BASE_URL = MODEL_CONFIGS[config_key]['base_url']

    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # 测试的模型列表 (取两个脚本的并集或常用集合)
    TESTED_MODEL_NAME_LIST = [
        # "gemini-2.5-pro",
        # "gpt-5",
        # "gpt-4o",
        # "claude-sonnet-4-5",
        # "gemini-2.5-flash",
        # "qwen3vl-235B",
        # "qwen3vl-30B",
        # "internvl3-5-241B",
        # "internvl3-5-30B",
        "step-3.5-flash"
    ] 
    
    # 更新后的工作目录
    BASE_DIR = '/path/to/project'
    LOG_DIR = BASE_DIR
    
    data_num = -1 
    MAX_CONCURRENT_CALLS = 10 
    
    print(f"--- Starting Batch Judgement for {len(TESTED_MODEL_NAME_LIST)} models ---")
    print(f"--- Data Directory: {BASE_DIR} ---")
    
    for tested_model_name in TESTED_MODEL_NAME_LIST:
        print(f"\nEvaluating: {tested_model_name}")
        
        input_file = os.path.join(BASE_DIR, f"{tested_model_name}_offline.jsonl")
        output_file = os.path.join(BASE_DIR, f"{tested_model_name}_offline_judged.jsonl") 
        log_file = os.path.join(LOG_DIR, f"{tested_model_name}_judgement.log")
        
        log_lock = threading.Lock()
        
        run_judgement(
            input_path=input_file,
            output_path=output_file,
            log_dir=LOG_DIR,
            log_path=log_file,
            lock=log_lock,
            client=client,
            judge_model_name=JUDGE_MODEL_NAME,
            max_workers=MAX_CONCURRENT_CALLS,
            data_num=data_num
        )
        
    print("\nAll models processed.")