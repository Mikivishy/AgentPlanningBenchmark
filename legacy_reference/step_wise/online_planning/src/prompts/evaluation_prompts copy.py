"""
Evaluation prompts for different datasets.

This module contains dataset-specific evaluation prompts with error classification
frameworks for evaluating agent planning predictions.
"""

import json
from typing import List, Dict

from ..utils.dataset_loader import get_base_dataset_type


class EvaluationPrompts:
    """Dataset-specific evaluation prompts with error classification."""
    
    @staticmethod
    def get_error_definitions(dataset_name: str) -> str:
        """Get dataset-specific error definitions with examples."""
        
        # Get base dataset type for evaluation
        base_type = get_base_dataset_type(dataset_name)
        
        # Build dataset-specific format requirement
        format_requirement = ""
        if base_type in ['gaia', 'gta']:
            # Dialog format datasets
            format_requirement = """
**CRITICAL FORMAT REQUIREMENT:**
- The predicted step MUST follow the standard dialogue format: `{"role": "assistant", "thought": "...", "tool_calls": [...]}`
- If the predicted step is missing the `role` field, `thought` field, or `tool_calls` array structure, it is a format error
- **EXAMPLE OF FORMAT ERROR:** Predicted step is `{"type": "function", "function": {...}}` instead of `{"role": "assistant", "thought": "...", "tool_calls": [{"type": "function", "function": {...}}]}`
- Format errors should be marked as E3_CONSTRAINT_VIOLATION because the output format is an implicit constraint
- Even if the tool call itself is correct, missing the required dialogue structure is a violation
"""

        elif base_type in ['toolbench', 'skywork']:
            # Dialog format datasets
            format_requirement = """
**CRITICAL FORMAT REQUIREMENT:**
- The predicted step MUST follow the standard dialogue format: `{"role": "assistant", "content": "...", "tool_calls": [...]}`
- If the predicted step is missing the `role` field, `content` field, or `tool_calls` array structure, it is a format error
- **EXAMPLE OF FORMAT ERROR:** Predicted step is `{"type": "function", "function": {...}}` instead of `{"role": "assistant", "content": "...", "tool_calls": [{"type": "function", "function": {...}}]}`
- Format errors should be marked as E3_CONSTRAINT_VIOLATION because the output format is an implicit constraint
- Even if the tool call itself is correct, missing the required dialogue structure is a violation
"""
        elif base_type == 'framethinker':
            # FrameThinker format (dialog format with role and content, with <think> and <action> tags)
            format_requirement = """
**CRITICAL FORMAT REQUIREMENT:**
- The predicted step MUST follow the standard dialogue format: `{"role": "assistant", "content": "..."}`
- The `content` field MUST contain both `<think></think>` and `<action></action>` tags
- The `<think></think>` section contains the reasoning process
- The `<action></action>` section contains the actual action/tool call
- **EXAMPLE OF CORRECT FORMAT:** `{"role": "assistant", "content": "<think>reasoning here</think>\n\n<action>tool_call_here</action>"}`
- **EXAMPLE OF FORMAT ERROR:** Missing `role` field, missing `<think>` or `<action>` tags, or incorrect tag structure
- Format errors should be marked as E3_CONSTRAINT_VIOLATION because the output format is an implicit constraint
- Even if the content itself is correct, missing the required dialogue structure or tags is a violation
"""
        elif base_type == 'opencua':
            # OpenCUA format (with separate action field)
            format_requirement = """
**CRITICAL FORMAT REQUIREMENT:**
- The predicted step MUST follow the standard dialogue format: `{"role": "assistant", "content": "observation and thought", "action": "action description"}`
- The `role` field MUST be "assistant"
- The `content` field contains the observation and thought process
- The `action` field contains a detailed natural language description of the specific UI operation to perform
- OpenCUA action format: detailed sentences describing WHAT to do, WHERE (UI element), and WHY (purpose)
- Example: `{"role": "assistant", "content": "Observation: ... Thought: ...", "action": "Click on the 'Account' icon in the Google apps menu in Google Chrome."}`
- If the predicted step is missing the `role`, `content`, or `action` field, it is a format error
- If the action uses function call syntax like "click(element)" instead of natural language, it is a format error
- Format errors should be marked as E3_CONSTRAINT_VIOLATION because the output format is an implicit constraint
"""
        else:
            # Default/unknown format
            format_requirement = """
**CRITICAL FORMAT REQUIREMENT:**
- The predicted step MUST follow the expected format for this dataset
- If the predicted step deviates from the required structure, it is a format error
- Format errors should be marked as E3_CONSTRAINT_VIOLATION because the output format is an implicit constraint
"""
        
        common_intro = f"""
**Error Classification Framework for Online Next-1 Prediction:**

Each error type is boolean (True/False) indicating whether that specific error exists in the NEXT STEP prediction.
Remember: You are evaluating a SINGLE NEXT STEP, not a complete trajectory.
{format_requirement}
**CRITICAL EVALUATION STANDARDS (EXTREMELY STRICT):**
- **ZERO TOLERANCE:** If a step has ANY flaw, it is an error.
- **POTENTIAL ERRORS ARE ERRORS:** If a step *might* be wrong, it IS wrong. Do not give the benefit of the doubt.
- **STRICTEST INTERPRETATION:** If there is any ambiguity, assume the worst case interpretation.
- **NOT ABOUT PLANNING ABILITY:** Do not forgive errors just because the "intent" was good.
- **ANY DEVIATION IS AN ERROR:** Any deviation from a perfect, optimal, and safe step is an error.
- **RISK IS ERROR:** Any risky action is an error, even if it might work.
"""
        
        if base_type == 'gaia':
            return common_intro + """
**1. E1_GOAL_MISALIGNMENT:**
   - **Definition:** The next step deviates from or contradicts the current goal based on the query and previous steps.
   - **What to check:** Does the predicted step align with the main query objective? Does it follow the logical progression established by previous steps?
   - **Examples:**
     * Query asks "Find the author of paper X and their h-index", previous steps searched for the paper, next step should find author info, but instead searches for paper citations (wrong sub-goal)
     * Query requires fact-checking a claim, previous steps gathered evidence, next step should analyze evidence, but instead starts gathering unrelated data
     * Multi-step task where step 1 is complete, next step should move to step 2, but instead repeats step 1 with different parameters
   - **NOT an error:** Taking a different but equally valid approach to achieve the same goal (e.g., using Google Scholar instead of PubMed to find author info)

**2. E2_PREMATURE_CONCLUSION:**
   - **Definition:** The next step attempts to conclude or output results when necessary intermediate steps are still missing.
   - **CRITICAL:** Check if the step only provides text response without tool calls when it should actually execute tools to complete the task.
   - **What to check:** Are all required sub-tasks completed? Does the step try to answer before gathering all necessary information?
   - **Examples:**
     * Query requires "find author AND publication date", previous steps only found the author, next step outputs final answer without finding publication date
     * Query asks for comparison of A vs B, only analyzed A so far, next step concludes without analyzing B
     * Task requires generating a report file, but next step only asks user if they want the file without actually creating it
     * Task needs data verification, but next step outputs result without verification
   - **Common mistake:** Saying "next we will do X" or asking user preferences instead of actually executing the necessary tool to do X

**3. E3_CONSTRAINT_VIOLATION:**
   - **Definition:** The next step violates an explicit constraint from the query, previous steps, or the required output format.
   - **CRITICAL:** Check if the predicted step is missing the required dialogue format structure (role, thought, tool_calls).
   - **What to check:** Does the step respect all specified constraints (time range, data source, format, method, etc.)? Does it follow the required output structure?
   - **Examples:**
     * Query specifies "use only 2023 data", next step queries 2022 data
     * Query says "search only in academic databases", next step uses general web search
     * Query requires "exact match", next step uses fuzzy matching
     * Predicted step is `{"type": "function", ...}` instead of `{"role": "assistant", "thought": "...", "tool_calls": [{"type": "function", ...}]}`
     * Previous step established a constraint (e.g., "use English sources"), next step violates it (searches Chinese sources)
   - **Format violation:** Any predicted step that does not include `role`, `thought`, and `tool_calls` fields in the proper structure is a constraint violation

**4. E4_LOGIC_ERROR:**
   - **Definition:** The next step has logical flaws OR POTENTIAL logical risks - it requires data/results that haven't been obtained yet, ignores available data from previous steps, or the reasoning flow is illogical or risky.
   - **CRITICAL:** Check if the step claims to do data processing/extraction but actually uses pre-processed data that doesn't exist yet.
   - **What to check:** Does the step use non-existent data? Does it ignore already-obtained data? Is the reasoning flow logical?
   - **Examples:**
     * Next step analyzes file content, but the file hasn't been read in previous steps (missing prerequisite)
     * Previous step already obtained author name "John Doe", next step searches for author name again (redundant operation)
     * Step claims to "extract email from text" but tool parameters already contain extracted email "john@example.com" without any extraction operation
     * Step depends on result from API call that hasn't been made yet
     * Logical contradiction: previous step confirmed "X is true", next step proceeds assuming "X is false"
   - **Common mistake:** Step description says it will extract/process data, but the tool call uses processed data that hasn't been created in previous steps

**5. E5_TOOL_USE_ERROR:**
   - **Definition:** The next step misuses (or POTENTIALLY misuses) a tool's function or passes incorrect parameters **according to the tool's actual specification in the Available Tools list**.
   - **CRITICAL:** Evaluate tool usage based SOLELY on whether it violates the tool's documented interface, NOT by comparing to the reference trajectory.
   - **CRITICAL:** Parameter names and values can be different from the reference trajectory as long as they conform to the tool's specification.
   - **What to check:** Does the tool call match the tool's specification? Are parameter types correct? Are required parameters present?
   - **Examples:**
     * Tool specification says "search(query: str, max_results: int)", next step calls it with "search(query: str, max_results: str)" (wrong type)
     * Tool requires "file_path" parameter, next step omits it (missing required parameter)
     * Tool specification says "accepts text files only", next step passes an image file (wrong file type)
     * Tool has 3 parameters but next step provides 5 parameters (extra parameters)
   - **NOT an error:** 
     * Using different but equally valid parameter values than the reference (e.g., max_results=10 vs max_results=5)
     * Using different parameter names if both conform to the tool's specification
     * Using a different valid enum value than the reference (e.g., mode="strict" vs mode="fuzzy" if both are valid)
     * Any variation that complies with the tool's documented specification

**6. E6_HALLUCINATION_ERROR:**
   - **Definition:** The next step calls a non-existent tool or references non-existent data.
   - **CRITICAL:** Tool names in the predicted step must EXACTLY match the tool names listed in the "Available Tools" section. Even minor variations in spelling, capitalization, or word order indicate hallucination.
   - **What to check:** Does every tool called exist in the Available Tools list? Are tool names spelled exactly correctly?
   - **Examples:**
     * Available tools: ["web_search", "read_file"], predicted step calls "google_search" (non-existent tool)
     * Available tools: ["SearchDatabase"], predicted step calls "searchDatabase" (wrong capitalization)
     * Available tools: ["get_weather"], predicted step calls "get_weather_data" (wrong tool name)
     * Step references data from "previous API response" but no API was called in previous steps
     * Step uses variable that was never defined or returned by previous tool calls
"""
        
        elif base_type == 'gta':
            return common_intro + """
**1. E1_GOAL_MISALIGNMENT:**
   - **Definition:** The next step deviates from the current goal or task flow.
   - **What to check:** Does the step align with the task's workflow? Is it progressing toward completing the task?
   - **Examples:**
     * Task is "analyze sales data and create a report", currently analyzing data, next step should continue analysis or start report creation, but instead starts unrelated image processing
     * Task requires Excel automation, previous steps set up data, next step should manipulate Excel, but instead calls web scraping tools
     * Workflow is "read file → process → save", currently at processing stage, next step should continue processing or move to saving, but instead goes back to reading a different file
     * Task is data transformation, next step should transform data, but instead validates already-transformed data that doesn't exist yet
   - **NOT an error:** Using a different valid approach within the same goal (e.g., using pandas vs openpyxl for Excel operations)

**2. E2_PREMATURE_CONCLUSION:**
   - **Definition:** The next step tries to finalize when required intermediate steps are incomplete.
   - **CRITICAL:** Check if the step only provides text response without tool calls when it should actually execute tools to complete the task.
   - **What to check:** Are all task requirements fulfilled? Are there missing steps in the workflow?
   - **Examples:**
     * Query requires "create chart and save it", chart created, next step outputs completion message without saving the file
     * Task needs "data cleaning → analysis → visualization", only cleaning done, next step concludes without analysis and visualization
     * Query asks to "update database and generate report", database updated, next step finishes without generating the report
     * Task requires generating a report file, but next step only asks user if they want the file without actually creating it
   - **Common mistake:** Saying "next we will do X" or asking user preferences instead of actually executing the necessary tool to do X

**3. E3_CONSTRAINT_VIOLATION:**
   - **Definition:** The next step violates query constraints, format requirements, or the required output structure.
   - **CRITICAL:** Check if the predicted step is missing the required dialogue format structure (role, thought, tool_calls).
   - **What to check:** Does the step follow specified formats, data types, output requirements, and structural constraints?
   - **Examples:**
     * Query specifies "save as CSV format", next step saves as JSON or Excel format
     * Query requires "use sheet named 'Sales'", next step operates on sheet named 'Revenue'
     * Query says "preserve original column order", next step reorders columns
     * Predicted step is `{"type": "function", ...}` instead of `{"role": "assistant", "thought": "...", "tool_calls": [{"type": "function", ...}]}`
     * Task specifies "round to 2 decimal places", next step uses 4 decimal places
   - **Format violation:** Any predicted step that does not include `role`, `thought`, and `tool_calls` fields is a constraint violation

**4. E4_LOGIC_ERROR:**
   - **Definition:** The next step has logical flaws OR POTENTIAL logical risks - missing prerequisites, redundant operations, or illogical reasoning flow.
   - **CRITICAL:** Check if the step claims to do data processing/extraction but actually uses pre-processed data that doesn't exist yet.
   - **What to check:** Are all prerequisites met? Is there redundancy? Is the logic sound?
   - **Examples:**
     * Next step updates Excel file rows, but the Excel file hasn't been created in previous steps (missing prerequisite)
     * Previous step already loaded data into DataFrame, next step loads the same data again (redundant)
     * Step says "extract account numbers from text" but tool parameters already contain extracted numbers like "XXXXXX0016" (illogical - using pre-processed data)
     * Step attempts to merge two datasets, but only one dataset has been loaded so far
     * Step tries to access column 'Revenue' that wasn't created in previous steps
   - **Common mistake:** Step description says it will extract/process data, but the tool call uses processed data that hasn't been created in previous steps

**5. E5_TOOL_USE_ERROR:**
   - **Definition:** The next step misuses (or POTENTIALLY misuses) tools or passes incorrect parameters **according to the tool's actual specification in the Available Tools list**.
   - **CRITICAL:** Evaluate tool usage based SOLELY on whether it violates the tool's documented interface, NOT by comparing to the reference trajectory.
   - **CRITICAL:** Parameter names and values can be different from the reference trajectory as long as they conform to the tool's specification.
   - **What to check:** Does the tool call match its specification? Are parameter types and values correct?
   - **Examples:**
     * Tool specification requires "sheet_name: str", next step passes a list of sheet names (wrong type)
     * Tool specification says "row_index: int", next step passes "row_index: '5'" as string (type mismatch)
     * Excel tool requires "cell_range: str" like "A1:B10", next step passes just "A1" when multiple cells are needed
     * Tool accepts specific enum values ["sum", "average", "count"], next step passes "total" (invalid value)
   - **NOT an error:** 
     * Using different but equally valid parameter values than the reference
     * Using different sheet names, ranges, or values that are equally valid
     * Using different aggregation functions if both are appropriate for the task
     * Any variation that complies with the tool's documented specification

**6. E6_HALLUCINATION_ERROR:**
   - **Definition:** The next step calls a non-existent tool or references non-existent data.
   - **CRITICAL:** Tool names in the predicted step must EXACTLY match the tool names listed in the "Available Tools" section. Even minor variations in spelling, capitalization, or word order indicate hallucination.
   - **What to check:** Does every tool exist in Available Tools? Are tool names exactly correct?
   - **Examples:**
     * Available tools: ["create_excel", "read_excel"], predicted step calls "open_excel" (non-existent tool)
     * Available tools: ["WriteToCell"], predicted step calls "write_to_cell" (wrong capitalization)
     * Available tools: ["update_dataframe"], predicted step calls "update_df" (abbreviated name not in tools list)
     * Step references "result from previous database query" but no database tool was called
     * Step uses column names that don't exist in the data structure
"""
        
        elif base_type == 'skywork':
            return common_intro + """
**1. E1_GOAL_MISALIGNMENT:**
   - **Definition:** The next step deviates from solving the math/reasoning problem.
   - **What to check:** Is the step working toward solving the mathematical or reasoning problem? Is it on the right track?
   - **Examples:**
     * Query asks to "solve the quadratic equation", next step searches for the history of quadratic equations instead of calculating
     * Problem requires calculating area of a shape, next step calculates perimeter instead (wrong mathematical goal)
     * Task is to find probability, next step does statistical mean calculation unrelated to probability
     * Multi-step math problem at step "find derivative", next step tries to find integral instead
     * Problem asks for numerical answer, next step provides symbolic manipulation that doesn't lead to numerical solution
   - **NOT an error:** Using a different valid mathematical approach (e.g., substitution method vs elimination method for solving equations)

**2. E2_PREMATURE_CONCLUSION:**
   - **Definition:** The next step outputs answer when verification or additional calculations are still needed.
   - **CRITICAL:** Check if the step only provides text response without tool calls when it should actually execute tools to complete the task.
   - **What to check:** Are all required calculations complete? Is verification needed but skipped?
   - **Examples:**
     * Query requires "calculate and verify the result", calculation done, next step outputs result without verification
     * Problem has multiple parts (a, b, c), only solved part (a), next step concludes without solving (b) and (c)
     * Problem asks to "solve and check your answer", solution found, next step outputs without checking
     * Task requires generating a report file, but next step only asks user if they want the file without actually creating it
     * Calculation done but rounding/formatting requirement not applied yet
   - **Common mistake:** Saying "next we will do X" or asking user preferences instead of actually executing the necessary tool to do X

**3. E3_CONSTRAINT_VIOLATION:**
   - **Definition:** The next step violates specified constraints or required output format.
   - **CRITICAL:** Check if the predicted step is missing the required dialogue format structure (role, content, tool_calls).
   - **What to check:** Are precision/format/unit requirements met? Does it follow the required structure?
   - **Examples:**
     * Query specifies "round to 2 decimal places", next step outputs 4 decimal places (3.1416 instead of 3.14)
     * Problem requires "answer in meters", next step gives answer in centimeters without conversion
     * Query says "express as a fraction", next step gives decimal answer
     * Query requires "show intermediate steps", next step only shows final answer
     * Predicted step is `{"type": "function", ...}` instead of `{"role": "assistant", "content": "...", "tool_calls": [{"type": "function", ...}]}`
   - **Format violation:** Any predicted step that does not include `role`, `content`, and `tool_calls` fields is a constraint violation

**4. E4_LOGIC_ERROR:**
   - **Definition:** The next step has logical flaws OR POTENTIAL logical risks - uses results not yet calculated, ignores available results, or has illogical reasoning.
   - **CRITICAL:** Check if the step claims to do data processing/extraction but actually uses pre-processed data that doesn't exist yet.
   - **What to check:** Are calculations done in correct order? Is there circular reasoning? Are intermediate results used correctly?
   - **Examples:**
     * Next step uses value of X in calculation, but X hasn't been calculated yet in previous steps
     * Previous step calculated "sum = 100", next step recalculates the same sum instead of using the result
     * Step claims to "derive the formula" but tool parameters already contain the derived formula without derivation
     * Calculation requires result from step A to compute step B, but step A hasn't been executed
     * Mathematical logic error: step assumes X > 0 but previous steps proved X < 0
   - **Common mistake:** Step description says it will extract/process data, but the tool call uses processed data that hasn't been created in previous steps

**5. E5_TOOL_USE_ERROR:**
   - **Definition:** The next step misuses (or POTENTIALLY misuses) tools (e.g., wrong data types) **according to the tool's actual specification in the Available Tools list**.
   - **CRITICAL:** Evaluate tool usage based SOLELY on whether it violates the tool's documented interface, NOT by comparing to the reference trajectory.
   - **CRITICAL:** Parameter names and values can be different from the reference trajectory as long as they conform to the tool's specification.
   - **What to check:** Are parameter types correct? Are mathematical operations valid for the tool?
   - **Examples:**
     * Tool specification says "calculate(expression: str)", next step passes a number instead of string (wrong type)
     * Tool requires "numbers: List[float]", next step passes a single float value (wrong structure)
     * Mathematical tool expects numerical input, next step passes string "abc" (invalid input type)
     * Tool specification has parameter range [0, 100], next step passes -5 (out of range)
   - **NOT an error:** 
     * Using different but equally valid parameter values than the reference
     * Using different mathematical expressions that achieve the same goal
     * Using different precision levels if both are within tool's specification
     * Any variation that complies with the tool's documented specification

**6. E6_HALLUCINATION_ERROR:**
   - **Definition:** The next step calls a non-existent tool.
   - **CRITICAL:** Tool names in the predicted step must EXACTLY match the tool names listed in the "Available Tools" section. Even minor variations in spelling, capitalization, or word order indicate hallucination.
   - **What to check:** Does the tool exist in Available Tools list? Is the name exactly correct?
   - **Examples:**
     * Available tools: ["calculator", "solver"], predicted step calls "calculate" (non-existent tool)
     * Available tools: ["MathSolver"], predicted step calls "mathsolver" (wrong capitalization)
     * Available tools: ["derive"], predicted step calls "differentiate" (different tool name)
     * Step references "previous calculation result" but no calculation tool was used in previous steps
"""
        
        elif base_type == 'toolbench':
            return common_intro + """
**1. E1_GOAL_MISALIGNMENT:**
   - **Definition:** The next step deviates from the API query intent.
   - **What to check:** Does the API call match the query's purpose? Is it fetching the right type of data?
   - **Examples:**
     * Query asks for "current weather forecast", next step calls historical weather data API (wrong time frame)
     * Query needs "user profile information", next step fetches user's purchase history (wrong data type)
     * Task is to "search for restaurants", next step calls hotel search API (wrong category)
     * Query requires "create a new resource", next step calls "update existing resource" API (wrong operation)
     * Multi-step API workflow: should call authentication API, but instead directly calls data API without auth
   - **NOT an error:** Using a different but equally valid API endpoint that serves the same purpose

**2. E2_PREMATURE_CONCLUSION:**
   - **Definition:** The next step returns results when required API calls are incomplete.
   - **CRITICAL:** Check if the step only provides text response without tool calls when it should actually execute tools to complete the task.
   - **What to check:** Are all required API calls completed? Is the data pipeline complete?
   - **Examples:**
     * Query needs "authenticate then fetch data", authentication done, next step returns without fetching data
     * Task requires "get list of IDs, then fetch details for each ID", got list, next step concludes without fetching details
     * Query asks to "create resource and verify creation", resource created, next step finishes without verification call
     * Task requires generating a report file, but next step only asks user if they want the file without actually creating it
     * API workflow needs "post data then get confirmation", only posted, next step ends without getting confirmation
   - **Common mistake:** Saying "next we will do X" or asking user preferences instead of actually executing the necessary tool to do X

**3. E3_CONSTRAINT_VIOLATION:**
   - **Definition:** The next step violates API constraints, requirements, or the required output format.
   - **CRITICAL:** Check if the predicted step is missing the required dialogue format structure (role, content, tool_calls).
   - **What to check:** Does the API call respect method, parameter, header, and format constraints?
   - **Examples:**
     * Query specifies "use GET request only", next step uses POST request (wrong HTTP method)
     * API requires specific header "API-Version: 2.0", next step omits this header (missing required header)
     * Query says "fetch maximum 10 items", next step requests 50 items (violates limit constraint)
     * API requires authentication token in header, next step puts it in query parameters (wrong location)
     * Predicted step is `{"type": "function", ...}` instead of `{"role": "assistant", "content": "...", "tool_calls": [{"type": "function", ...}]}`
   - **Format violation:** Any predicted step that does not include `role`, `content`, and `tool_calls` fields is a constraint violation

**4. E4_LOGIC_ERROR:**
   - **Definition:** The next step has logical flaws OR POTENTIAL logical risks - makes API calls without required prerequisites, has wrong dependencies, or illogical reasoning flow.
   - **CRITICAL:** Check if the step claims to do data processing/extraction but actually uses pre-processed data that doesn't exist yet.
   - **What to check:** Are API prerequisites met? Is the call sequence logical? Are dependencies satisfied?
   - **Examples:**
     * Next step calls data API that requires authentication token, but authentication hasn't been done in previous steps (missing prerequisite)
     * Previous step got list of user IDs [1,2,3], next step requests details for ID 5 that wasn't in the list (illogical)
     * Step claims to "extract user emails from API response" but tool parameters already contain extracted emails without any extraction (using pre-processed data)
     * API call depends on result from previous API call that failed or wasn't made
     * Step tries to update a resource before creating it
   - **Common mistake:** Step description says it will extract/process data, but the tool call uses processed data that hasn't been created in previous steps

**5. E5_TOOL_USE_ERROR:**
   - **Definition:** The next step uses incorrect (or POTENTIALLY incorrect) API parameters or data types **according to the API/tool's actual specification in the Available Tools list**.
   - **CRITICAL:** Evaluate tool usage based SOLELY on whether it violates the tool's documented interface, NOT by comparing to the reference trajectory.
   - **CRITICAL:** Check if tool parameters use vague language descriptions instead of concrete, specific values that fulfill the task requirements.
   - **CRITICAL:** Parameter names and values can be different from the reference trajectory as long as they conform to the tool's specification and are sufficiently specific.
   - **What to check:** Are parameter types correct? Are required parameters present? Are parameter values specific enough?
   - **Examples:**
     * API specification requires "user_id: int", next step passes "user_id: '123'" as string (wrong type)
     * API requires parameters {user_id, auth_token}, next step only provides {user_id} (missing required parameter)
     * API expects enum ["json", "xml"], next step passes "yaml" (invalid value)
     * Task requires statistical data about database, but predicted step uses vague descriptions like "Defined for each table" instead of concrete numbers (insufficiently specific)
     * API specification requires date format "YYYY-MM-DD", next step passes "01/15/2024" (wrong format)
   - **NOT an error:** 
     * Using different but equally valid parameter values than the reference
     * Using different API endpoints that achieve the same result
     * Using different request formats if both are supported by the API
     * Using different specific values (e.g., limit=10 vs limit=20) as long as both conform to specification and are sufficiently specific
     * Any variation that complies with the tool's documented specification and provides concrete, specific values

**6. E6_HALLUCINATION_ERROR:**
   - **Definition:** The next step calls non-existent API endpoints or tools.
   - **CRITICAL:** Tool names in the predicted step must EXACTLY match the tool names listed in the "Available Tools" section. Even minor variations in spelling, capitalization, or word order indicate hallucination.
   - **What to check:** Does the API/tool exist? Is the endpoint name exactly correct?
   - **Examples:**
     * Available tools: ["get_user_data", "update_user"], predicted step calls "fetch_user" (non-existent API)
     * Available tools: ["WeatherAPI"], predicted step calls "weatherAPI" (wrong capitalization)
     * Available tools: ["search_v2"], predicted step calls "search" (missing version suffix)
     * Step references "response from payment API" but no payment API exists in available tools
     * Step calls endpoint "/api/v3/users" but only "/api/v2/users" exists in available tools
"""
        
        elif base_type == 'framethinker':
            return common_intro + """
**1. E1_GOAL_MISALIGNMENT:**
   - **Definition:** The next step deviates from the reasoning task's goal.
   - **What to check:** Is the reasoning step aligned with the logical goal? Does it progress the reasoning chain correctly?
   - **Examples:**
     * Query requires logical deduction to prove a theorem, next step performs pattern matching on examples instead
     * Task is to evaluate validity of an argument, next step searches for counterexamples when should analyze argument structure
     * Reasoning goal is to establish causation, next step only shows correlation
     * Task requires inductive reasoning from specific to general, next step uses deductive reasoning from general to specific
     * Multi-step proof at "establish premise 2", next step jumps to conclusion without establishing premise 2
   - **NOT an error:** Using a different but equally valid reasoning approach (e.g., proof by contradiction vs direct proof)

**2. E2_PREMATURE_CONCLUSION:**
   - **Definition:** The next step jumps to conclusion when required reasoning steps are missing.
   - **CRITICAL:** Check if the step only provides text response without tool calls when it should actually execute tools to complete the task.
   - **What to check:** Are all necessary premises established? Are intermediate reasoning steps complete?
   - **Examples:**
     * Query needs "establish premises → make inferences → draw conclusion", premises established, next step jumps to conclusion without inferences
     * Task requires validating each step of an argument (steps 1-5), only validated steps 1-2, next step concludes argument is valid
     * Reasoning requires "analyze assumption → test assumption → conclude", assumption analyzed, next step concludes without testing
     * Task requires generating a report file, but next step only asks user if they want the file without actually creating it
     * Proof needs steps A → B → C, completed A → B, next step claims proof complete without establishing C
   - **Common mistake:** Saying "next we will do X" or asking user preferences instead of actually executing the necessary tool to do X

**3. E3_CONSTRAINT_VIOLATION:**
   - **Definition:** The next step violates reasoning constraints specified in the query or the required output format.
   - **CRITICAL:** Check if the predicted step is missing the required dialogue format structure (role, content) and the required <think> and <action> tags.
   - **What to check:** Are reasoning constraints respected? Is the required format followed? Does the content include both <think></think> and <action></action> tags?
   - **Examples:**
     * Query says "use only given facts, no external knowledge", next step introduces external assumptions
     * Task specifies "use formal logic notation", next step uses informal natural language
     * Query requires "provide counterexample if exists", next step gives general argument without concrete counterexample
     * Predicted step is `{"name": "...", "reason": "..."}` instead of `{"role": "assistant", "content": "<think>...</think>\n\n<action>...</action>"}`
     * Content is missing <think> or <action> tags, or tags are incorrectly structured
     * Task says "use deductive reasoning only", next step uses inductive generalization
   - **Format violation:** Any predicted step that does not include `role` and `content` fields, or missing <think></think> and <action></action> tags in content, is a constraint violation

**4. E4_LOGIC_ERROR:**
   - **Definition:** The next step has logical flaws OR POTENTIAL logical risks - uses conclusions before establishing supporting premises, redundantly re-establishes premises, or has illogical reasoning.
   - **CRITICAL:** Check if the step claims to do data processing/extraction but actually uses pre-processed data that doesn't exist yet.
   - **What to check:** Is the reasoning logically sound? Are premises used before they're established? Is there circular reasoning?
   - **Examples:**
     * Next step uses conclusion "X is true" in its reasoning, but "X is true" hasn't been proven in previous steps (using unproven conclusion)
     * Previous step established premise "All A are B", next step re-establishes "All A are B" instead of using it (redundant)
     * Step claims to "derive theorem from axioms" but the derivation already contains the theorem without actual derivation steps (illogical)
     * Circular reasoning: step uses P to prove Q, but Q was used to establish P earlier
     * Step assumes "if P then Q" is equivalent to "if Q then P" (logical fallacy)
   - **Common mistake:** Step description says it will extract/process data, but the tool call uses processed data that hasn't been created in previous steps

**5. E5_TOOL_USE_ERROR:**
   - **Definition:** The next step misuses (or POTENTIALLY misuses) reasoning tools or methods **according to the tool's actual specification in the Available Tools list**.
   - **CRITICAL:** Evaluate tool usage based SOLELY on whether it violates the tool's documented interface, NOT by comparing to the reference trajectory.
   - **CRITICAL:** Parameter names and values can be different from the reference trajectory as long as they conform to the tool's specification.
   - **What to check:** Is the reasoning tool used correctly? Are parameters appropriate for the tool?
   - **Examples:**
     * Tool specification says "deductive_reasoning(premises: List[str], conclusion: str)", next step passes conclusion as a list (wrong type)
     * Reasoning tool requires "formal_logic_notation", next step provides natural language (wrong format)
     * Tool is for "syllogistic reasoning" requiring exactly 2 premises, next step provides 3 premises (wrong parameter count)
     * Tool specification says "evaluate_argument(argument: str, type: ['valid', 'sound'])", next step uses type: 'correct' (invalid enum value)
   - **NOT an error:** 
     * Using different but equally valid parameter values than the reference
     * Using different reasoning approaches if both conform to the tool specification
     * Using different formal notations if both are accepted by the tool
     * Any variation that complies with the tool's documented specification

**6. E6_HALLUCINATION_ERROR:**
   - **Definition:** The next step uses non-existent reasoning methods or tools.
   - **CRITICAL:** Tool names in the predicted step must EXACTLY match the tool names listed in the "Available Tools" section. Even minor variations in spelling, capitalization, or word order indicate hallucination.
   - **What to check:** Does the reasoning tool exist? Is the tool name exactly correct?
   - **Examples:**
     * Available tools: ["deductive_reasoning", "inductive_reasoning"], predicted step calls "abductive_reasoning" (non-existent tool)
     * Available tools: ["LogicProver"], predicted step calls "logicprover" (wrong capitalization)
     * Available tools: ["verify_proof"], predicted step calls "check_proof" (different tool name)
     * Step references "result from previous inference" but no inference tool was called in previous steps
     * Step uses reasoning method that doesn't exist in available tools
"""
        
        else:
            return common_intro + """
**1. E1_GOAL_MISALIGNMENT:** The next step deviates from the current goal.
**2. E2_PREMATURE_CONCLUSION:** The next step concludes too early.
**3. E3_CONSTRAINT_VIOLATION:** The next step violates constraints or the required output format for this dataset.
**4. E4_LOGIC_ERROR:** The next step has logical flaws or wrong dependencies.
**5. E5_TOOL_USE_ERROR:** The next step misuses tools according to their documented specifications (NOT by comparing to reference).
**6. E6_HALLUCINATION_ERROR:** The next step uses non-existent tools.
"""
    
    @staticmethod
    def get_scoring_rubric() -> str:
        """Get scoring rubric for all datasets."""
        return """
**Scoring Levels (with examples):**

- **1.0 (Perfect):** No errors. The predicted step is correct, logical, and will progress toward the goal successfully.
  - Example: Correctly identifies a valid next step with proper tool usage that moves the task forward.
  - Requirement: Zero error flags (E1-E6 all False).
  - Is Correct: YES
  - **Note:** The step does NOT need to be "optimal" or "most efficient" - it just needs to be correct and reasonable.

- **0.8 (Very Good):** At most ONE very minor issue (e.g., only E5 with trivial impact) that doesn't affect correctness.
  - Example: Uses slightly different but equally valid search terms, or chooses a different valid page range.
  - Requirement: At most one minor error flag, and it must be E5 only with minimal impact.
  - Is Correct: NO (any error present)

- **0.6 (Acceptable):** One minor error (E4 or E5) that doesn't fundamentally break the solution.
  - Example: Minor parameter issue or small dependency oversight that can be easily recovered in subsequent steps.
  - Requirement: Only one error flag, and it must be E4 or E5.
  - Is Correct: NO (any error present)

- **0.4 (Problematic):** Multiple minor errors (2+ of E4/E5) OR one moderate error (E2).
  - Example: Skips a necessary step and has tool usage issues.
  - Requirement: Two or more minor error flags OR one E2 flag.
  - Is Correct: NO

- **0.2 (Severe):** Major errors (E1, E3, or E6) OR multiple moderate errors.
  - Example: Goal misalignment or calling non-existent tools.
  - Requirement: At least one major error flag (E1/E3/E6) OR multiple E2/E4/E5 flags.
  - Is Correct: NO

- **0.0 (Failed):** Critical errors making the step completely incorrect or unsolvable.
  - Example: Complete misunderstanding of the task combined with multiple errors.
  - Requirement: Multiple major error flags OR complete logical breakdown.
  - Is Correct: NO

**EVALUATION PRINCIPLES:** 
- Evaluate based on **CORRECTNESS and REASONABLENESS**, not optimality or efficiency
- **DO NOT** penalize for being "suboptimal" if the approach is valid and will work
- **DO NOT** compare with reference to judge "better or worse" - only check if prediction is valid
- **REFERENCE USAGE:** Use the reference trajectory ONLY to understand the correct logic and direction. Do not penalize for valid deviations.
- Different valid approaches should NOT be penalized even if they differ from the reference
- The reference is just ONE possible correct solution, not the only correct one
- A step is correct if it: (1) aligns with the goal, (2) uses tools properly, (3) has sound logic, (4) will successfully progress the task
- Only mark errors when there are actual problems (wrong tools, missing data, logical flaws, violations)
- **CRITICAL: ANY error (E1-E6) automatically means is_correct = NO. ONLY perfect predictions with zero errors can be marked as correct.**
"""
    
    @staticmethod
    def build_evaluation_prompt(
        dataset_name: str,
        query: str,
        trajectory_prefix: str,
        predicted_step: str,
        reference_next_step: str,
        reference_remaining_steps: str,
        tools: List,  # Can be List[Dict] or List[str]
        prefix_file_count: int = 0  # For opencua: number of images test model saw
    ) -> str:
        """Build complete evaluation prompt."""
        
        error_defs = EvaluationPrompts.get_error_definitions(dataset_name)
        scoring = EvaluationPrompts.get_scoring_rubric()
        
        # Build tools information
        # Note: If tools is an empty list, tool information is in the system_prompt
        tools_info = ""
        if tools:  # Only add tools section if tools list is not empty
            tools_info = "\n**Available Tools:**\n"
            for tool in tools:
                # Handle both dict format (with name/description) and string format
                if isinstance(tool, dict):
                    # Check for skywork nested format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
                    if 'function' in tool and isinstance(tool['function'], dict):
                        # Skywork format: extract from nested function object
                        func = tool['function']
                        name = func.get('name', 'unknown')
                        # Combine description and parameters as JSON string
                        desc_dict = {
                            'description': func.get('description', ''),
                            'parameters': func.get('parameters', {})
                        }
                        desc = json.dumps(desc_dict, ensure_ascii=False)
                        tools_info += f"- {name}: {desc}\n"
                    # Check for framethinker format: {"name": ..., "description": ..., "example": ...}
                    elif 'name' in tool and 'example' in tool:
                        # Framethinker format: combine description and example
                        name = tool.get('name', 'unknown')
                        desc = tool.get('description', '')
                        example = tool.get('example', '')
                        # Merge description and example
                        if example:
                            combined_desc = f"{desc} Example: {example}" if desc else f"Example: {example}"
                        else:
                            combined_desc = desc if desc else 'No description'
                        tools_info += f"- {name}: {combined_desc}\n"
                    else:
                        # Standard format: {"tool_name": ..., "description": ...}
                        name = tool.get('tool_name', 'unknown')
                        desc = tool.get('description', 'No description')
                        tools_info += f"- {name}: {desc}\n"
                else:
                    # For string format (e.g., opencua)
                    tools_info += f"- {tool}\n"
        else:
            # If tools list is empty, tool information is already in system_prompt
            tools_info = "\n**Note:** Available tools are defined in the system prompt.\n"
        
        # Build reference steps section
        reference_info = f"""**Reference Next Step:**
{reference_next_step}

**Guidance on Reference Trajectory:**
- The reference trajectory is provided to help you understand the correct direction and logic of the task.
- Use it to judge if the predicted step is factually and logically correct in the context of the task.
- DO NOT compare the predicted step with the reference step to decide which is "better".
- If the predicted step achieves the same goal as the reference step (or a valid alternative goal) using a valid method, it is correct."""
        
        if reference_remaining_steps and reference_remaining_steps != "[]" and reference_remaining_steps != "":
            reference_info += f"""

**Reference Remaining Steps (for context, not ground truth):**
{reference_remaining_steps}"""
        
        # For opencua, add information about image context
        image_context_info = ""
        if prefix_file_count > 0:
            image_context_info = f"""

**IMPORTANT - Image Context for OpenCUA:**
- The test model was shown the FIRST {prefix_file_count} images/screenshots (corresponding to the trajectory prefix)
- You are seeing ALL images (both prefix and remaining) to understand the full context
- Images {prefix_file_count + 1} onwards were NOT visible to the test model when making its prediction
- When evaluating, consider: Did the model make a reasonable prediction based on ONLY the first {prefix_file_count} images?
"""
        
        prompt = f"""You are evaluating a planning model's next step prediction.

**Dataset:** {dataset_name}

**User Query:**
{query}

**Trajectory So Far:**
{trajectory_prefix}
{tools_info}
{image_context_info}
{reference_info}

**Predicted Next Step:**
{predicted_step}

{error_defs}

{scoring}

**Your Task:**
Evaluate the SINGLE NEXT STEP prediction with EXTREME STRICTNESS.

**CRITICAL EVALUATION REQUIREMENTS:**
1. **POTENTIAL ERROR = ERROR:** If a step *might* fail or has *potential* risks, it is an ERROR.
2. **STRICTEST INTERPRETATION:** If there is any doubt, mark it as an error.
3. **IGNORE INTENT:** Do not evaluate "planning ability" or "intent". Evaluate the strict correctness of the step.
4. **OPTIMALITY:** Do NOT penalize suboptimal steps. If a step is valid and correct, it is NOT an error, even if it is less efficient than the reference.
5. **SAFETY FIRST:** Any ambiguity is an error.
6. **NO LENIENCY:** Do not be lenient. Be a harsh critic.

**CRITICAL FOR E2_PREMATURE_CONCLUSION:**
- **CHECK** if the predicted step has NO tool calls (empty tool_calls array or just text content)
- **BUT** the task clearly requires executing a tool to create/generate/process something
- **EXAMPLE:** Task requires "generate HTML report", reference calls generate_html_report tool, but prediction only says "let me show you the table" or asks "do you want to export?" without actually calling any tool
- This is premature conclusion because the step talks about completing the task without actually doing it
- Text-only responses are acceptable ONLY when the task is genuinely complete or user input is needed

**CRITICAL FOR E4_LOGIC_ERROR:**
- **CHECK** if the step's description/thought claims to do data extraction/processing/cleaning
- **BUT** the actual tool call uses already-processed data that doesn't exist in previous steps
- **EXAMPLE:** Step says "extract the last 4 digits of account numbers" but tool parameters already contain extracted data like "XXXXXX0016"
- This is a logic error because the step skips necessary intermediate processing or has illogical reasoning
- The processed data must come from a previous step or be created in THIS step with appropriate tools

**CRITICAL FOR E5_TOOL_USE_ERROR:**
- **STRICT PARAMETER CHECK:** Check if parameters are optimal and correct.
- **VAGUE PARAMETERS:** Any vague or non-specific parameter is an ERROR.
- **POTENTIAL MISUSE:** If the parameter value *might* be wrong or lead to unexpected results, it is an ERROR.
- **SPECIFICATION COMPLIANCE:** Must strictly adhere to tool specifications.

**CRITICAL FOR E6_HALLUCINATION_ERROR:**
- **MUST CHECK:** Verify that every tool name used in the predicted step exists in the "Available Tools" list
- Tool names must match EXACTLY - check spelling, capitalization, and word order
- Even slight variations in tool names (different spelling, word order, or capitalization) should be flagged as hallucination
- The tool name in the predicted step must be identical to one of the tool names listed in Available Tools

Provide:

1. Error classification (True/False for each E1-E6) - BE STRICT, mark True if ANY doubt exists
2. Score (0.0, 0.2, 0.4, 0.6, 0.8, or 1.0) - Apply the strict scoring rubric
3. Detailed reasoning explaining your evaluation and why you marked each error
4. Is correct? (YES ONLY if ALL error flags are False, otherwise NO)

**Output Format (JSON):**
```json
{{
    "E1_GOAL_MISALIGNMENT": false,
    "E2_PREMATURE_CONCLUSION": false,
    "E3_CONSTRAINT_VIOLATION": false,
    "E4_LOGIC_ERROR": false,
    "E5_TOOL_USE_ERROR": false,
    "E6_HALLUCINATION_ERROR": false,
    "score": 0.8,
    "reasoning": "The predicted next step correctly identifies...",
    "is_correct": "YES"
}}
```

Provide ONLY the JSON output, no additional text.
"""
        return prompt
    
    @staticmethod
    def build_evaluation_prompt_tool_broken(
        dataset_name: str,
        query: str,
        trajectory_prefix: str,
        broken_tool_name: str,
        replacement_tool: Dict,
        predicted_step: str,
        tools: List
    ) -> str:
        """Build evaluation prompt for tool broken scenario."""
        
        # Build tools information
        tools_info = ""
        if tools:
            tools_info = "\n**Available Tools (including replacement):**\n"
            for tool in tools:
                if isinstance(tool, dict):
                    if 'function' in tool and isinstance(tool['function'], dict):
                        func = tool['function']
                        name = func.get('name', 'unknown')
                        desc_dict = {
                            'description': func.get('description', ''),
                            'parameters': func.get('parameters', {})
                        }
                        desc = json.dumps(desc_dict, ensure_ascii=False)
                        tools_info += f"- {name}: {desc}\n"
                    elif 'name' in tool and 'example' in tool:
                        name = tool.get('name', 'unknown')
                        desc = tool.get('description', '')
                        example = tool.get('example', '')
                        combined_desc = f"{desc} Example: {example}" if desc else f"Example: {example}"
                        tools_info += f"- {name}: {combined_desc}\n"
                    else:
                        name = tool.get('tool_name', 'unknown')
                        desc = tool.get('description', 'No description')
                        tools_info += f"- {name}: {desc}\n"
                else:
                    tools_info += f"- {tool}\n"
        else:
            tools_info = "\n**Note:** Available tools are defined in the system prompt.\n"

        replacement_tool_name = replacement_tool.get('name')
        if not replacement_tool_name:
            replacement_tool_name = replacement_tool.get('tool_name')
        if not replacement_tool_name and 'function' in replacement_tool:
            replacement_tool_name = replacement_tool['function'].get('name')

        prompt = f"""You are evaluating a planning model's response to a "Tool Broken" scenario.

**Dataset:** {dataset_name}

**User Query:**
{query}

**Scenario:**
The model attempted to use the tool `{broken_tool_name}`, but it failed with an error message indicating the tool is broken.
A replacement tool `{replacement_tool_name}` has been made available to the model.
The model should ideally:
1. Recognize the failure.
2. Use the replacement tool `{replacement_tool_name}` OR find another valid alternative strategy using available tools.

**Trajectory So Far (ending with the error):**
{trajectory_prefix}
{tools_info}

**Predicted Next Step:**
{predicted_step}

**Evaluation Criteria:**
1. **Tool Selection:** Did the model select the replacement tool `{replacement_tool_name}`? 
   - OR did it select another *valid* tool that can solve the problem? (e.g. using a search engine if specific tool fails)
   - OR did it provide a valid reasoning for why it cannot proceed?
2. **Correctness:** Is the tool usage correct (parameters, syntax)?
3. **Goal Alignment:** Does the step move towards solving the original user query?

**Scoring:**
- **1.0 (Correct):** Model successfully used the replacement tool OR a valid alternative strategy.
- **0.0 (Incorrect):** Model tried to use the broken tool again, used a non-existent tool, or hallucinated.

**Output Format (JSON):**
```json
{{
    "score": 1.0,
    "is_correct": true,
    "reasoning": "The model correctly identified that X is broken and used replacement tool Y..."
}}
```
Provide ONLY the JSON output.
"""
        return prompt

    @staticmethod
    def build_evaluation_prompt_next2(
        dataset_name: str,
        query: str,
        trajectory_prefix: str,
        predicted_steps: str,  # Complete prediction output (should be JSON array with 2 steps)
        ground_truth_steps: str,  # Complete ground truth (should be JSON array with 2 steps)
        reference_remaining_steps: str,
        tools: List,
        prefix_file_count: int = 0
    ) -> str:
        """Build evaluation prompt for next2 (evaluating two steps together)."""
        
        error_defs = EvaluationPrompts.get_error_definitions(dataset_name)
        scoring = EvaluationPrompts.get_scoring_rubric()
        
        # Build tools information
        tools_info = ""
        if tools:
            tools_info = "\n**Available Tools:**\n"
            for tool in tools:
                if isinstance(tool, dict):
                    if 'function' in tool and isinstance(tool['function'], dict):
                        func = tool['function']
                        name = func.get('name', 'unknown')
                        desc_dict = {
                            'description': func.get('description', ''),
                            'parameters': func.get('parameters', {})
                        }
                        desc = json.dumps(desc_dict, ensure_ascii=False)
                        tools_info += f"- {name}: {desc}\n"
                    elif 'name' in tool and 'example' in tool:
                        name = tool.get('name', 'unknown')
                        desc = tool.get('description', '')
                        example = tool.get('example', '')
                        combined_desc = f"{desc} Example: {example}" if desc else f"Example: {example}"
                        tools_info += f"- {name}: {combined_desc}\n"
                    else:
                        name = tool.get('tool_name', 'unknown')
                        desc = tool.get('description', 'No description')
                        tools_info += f"- {name}: {desc}\n"
                else:
                    tools_info += f"- {tool}\n"
        else:
            tools_info = "\n**Note:** Available tools are defined in the system prompt.\n"
        
        # Build reference steps section
        reference_info = f"""**Reference Next Two Steps:**

{ground_truth_steps}

**Guidance on Reference Trajectory:**
- The reference trajectory is provided to help you understand the correct direction and logic of the task.
- Use it to judge if the predicted steps are factually and logically correct in the context of the task.
- DO NOT compare the predicted steps with the reference steps to decide which is "better".
- If the predicted steps achieve the same goal as the reference steps (or a valid alternative goal) using a valid method, they are correct."""
        
        if reference_remaining_steps and reference_remaining_steps != "[]" and reference_remaining_steps != "":
            reference_info += f"""

**Reference Remaining Steps (for context, not ground truth):**
{reference_remaining_steps}"""
        
        # For opencua, add information about image context
        image_context_info = ""
        if prefix_file_count > 0:
            image_context_info = f"""

**IMPORTANT - Image Context for OpenCUA:**
- The test model was shown the FIRST {prefix_file_count} images/screenshots (corresponding to the trajectory prefix)
- You are seeing ALL images (both prefix and remaining) to understand the full context
- Images {prefix_file_count + 1} onwards were NOT visible to the test model when making its prediction
- When evaluating, consider: Did the model make reasonable predictions based on ONLY the first {prefix_file_count} images?
"""
        
        prompt = f"""You are evaluating a planning model's next TWO steps prediction.

**Dataset:** {dataset_name}

**User Query:**
{query}

**Trajectory So Far:**
{trajectory_prefix}
{tools_info}
{image_context_info}
{reference_info}

**Predicted Next Two Steps:**
{predicted_steps}

**Note:** The prediction above should contain TWO steps. Please parse and evaluate both steps.

{error_defs}

{scoring}

**Your Task:**
Evaluate the TWO-STEP SEQUENCE prediction with EXTREME STRICTNESS.

**CRITICAL EVALUATION REQUIREMENTS:**
1. **POTENTIAL ERROR = ERROR:** If ANY step in the sequence *might* fail or has *potential* risks, it is an ERROR.
2. **STRICTEST INTERPRETATION:** If there is any doubt, mark it as an error.
3. **IGNORE INTENT:** Do not evaluate "planning ability" or "intent". Evaluate the strict correctness of the steps.
4. **OPTIMALITY:** Do NOT penalize suboptimal steps. If a step is valid and correct, it is NOT an error, even if it is less efficient than the reference.
5. **SAFETY FIRST:** Any ambiguity is an error.
6. **NO LENIENCY:** Do not be lenient. Be a harsh critic.
7. **SEQUENCE COHERENCE:** Check if Step 2 logically follows Step 1. If the sequence is disjointed or Step 2 ignores the result of Step 1, it is an error.

**CRITICAL FOR E2_PREMATURE_CONCLUSION:**
- **CHECK** if ANY step in the sequence has NO tool calls (empty tool_calls array or just text content)
- **BUT** the task clearly requires executing a tool to create/generate/process something
- **EXAMPLE:** Task requires "generate HTML report", reference calls generate_html_report tool, but prediction only says "let me show you the table" or asks "do you want to export?" without actually calling any tool
- This is premature conclusion because the step talks about completing the task without actually doing it
- Text-only responses are acceptable ONLY when the task is genuinely complete or user input is needed

**CRITICAL FOR E4_LOGIC_ERROR:**
- **CHECK** if ANY step's description/thought claims to do data extraction/processing/cleaning
- **BUT** the actual tool call uses already-processed data that doesn't exist in previous steps
- **EXAMPLE:** Step says "extract the last 4 digits of account numbers" but tool parameters already contain extracted data like "XXXXXX0016"
- This is a logic error because the step skips necessary intermediate processing or has illogical reasoning
- The processed data must come from a previous step or be created in THIS step with appropriate tools
- **DEPENDENCY CHECK:** If Step 2 uses data that Step 1 is *supposed* to generate, ensure Step 1 actually generates it.

**CRITICAL FOR E5_TOOL_USE_ERROR:**
- **STRICT PARAMETER CHECK:** Check if parameters are optimal and correct for ALL steps.
- **VAGUE PARAMETERS:** Any vague or non-specific parameter is an ERROR.
- **POTENTIAL MISUSE:** If the parameter value *might* be wrong or lead to unexpected results, it is an ERROR.
- **SPECIFICATION COMPLIANCE:** Must strictly adhere to tool specifications.

**CRITICAL FOR E6_HALLUCINATION_ERROR:**
- **MUST CHECK:** Verify that every tool name used in ALL predicted steps exists in the "Available Tools" list
- Tool names must match EXACTLY - check spelling, capitalization, and word order
- Even slight variations in tool names (different spelling, word order, or capitalization) should be flagged as hallucination
- The tool name in the predicted step must be identical to one of the tool names listed in Available Tools

Provide:

1. Error classification (True/False for each E1-E6) - BE STRICT, mark True if ANY doubt exists in ANY step
2. Score (0.0, 0.2, 0.4, 0.6, 0.8, or 1.0) - Apply the strict scoring rubric
3. Detailed reasoning explaining your evaluation and why you marked each error
4. Is correct? (YES ONLY if ALL error flags are False, otherwise NO)

**Output Format (JSON):**
```json
{{
    "E1_GOAL_MISALIGNMENT": false,
    "E2_PREMATURE_CONCLUSION": false,
    "E3_CONSTRAINT_VIOLATION": false,
    "E4_LOGIC_ERROR": false,
    "E5_TOOL_USE_ERROR": false,
    "E6_HALLUCINATION_ERROR": false,
    "score": 0.8,
    "reasoning": "The predicted two-step sequence correctly identifies...",
    "is_correct": "YES"
}}
```

Provide ONLY the JSON output, no additional text.
"""
        return prompt
    
    @staticmethod
    def build_evaluation_prompt_next3(
        dataset_name: str,
        query: str,
        trajectory_prefix: str,
        predicted_steps: str,  # Complete prediction output (should be JSON array with 3 steps)
        ground_truth_steps: str,  # Complete ground truth (should be JSON array with 3 steps)
        reference_remaining_steps: str,
        tools: List,
        prefix_file_count: int = 0
    ) -> str:
        """Build evaluation prompt for next3 (evaluating three steps together)."""
        
        error_defs = EvaluationPrompts.get_error_definitions(dataset_name)
        scoring = EvaluationPrompts.get_scoring_rubric()
        
        # Build tools information
        tools_info = ""
        if tools:
            tools_info = "\n**Available Tools:**\n"
            for tool in tools:
                if isinstance(tool, dict):
                    if 'function' in tool and isinstance(tool['function'], dict):
                        func = tool['function']
                        name = func.get('name', 'unknown')
                        desc_dict = {
                            'description': func.get('description', ''),
                            'parameters': func.get('parameters', {})
                        }
                        desc = json.dumps(desc_dict, ensure_ascii=False)
                        tools_info += f"- {name}: {desc}\n"
                    elif 'name' in tool and 'example' in tool:
                        name = tool.get('name', 'unknown')
                        desc = tool.get('description', '')
                        example = tool.get('example', '')
                        combined_desc = f"{desc} Example: {example}" if desc else f"Example: {example}"
                        tools_info += f"- {name}: {combined_desc}\n"
                    else:
                        name = tool.get('tool_name', 'unknown')
                        desc = tool.get('description', 'No description')
                        tools_info += f"- {name}: {desc}\n"
                else:
                    tools_info += f"- {tool}\n"
        else:
            tools_info = "\n**Note:** Available tools are defined in the system prompt.\n"
        
        # Build reference steps section
        reference_info = f"""**Reference Next Three Steps:**

{ground_truth_steps}

**Guidance on Reference Trajectory:**
- The reference trajectory is provided to help you understand the correct direction and logic of the task.
- Use it to judge if the predicted steps are factually and logically correct in the context of the task.
- DO NOT compare the predicted steps with the reference steps to decide which is "better".
- If the predicted steps achieve the same goal as the reference steps (or a valid alternative goal) using a valid method, they are correct."""
        
        if reference_remaining_steps and reference_remaining_steps != "[]" and reference_remaining_steps != "":
            reference_info += f"""

**Reference Remaining Steps (for context, not ground truth):**
{reference_remaining_steps}"""
        
        # For opencua, add information about image context
        image_context_info = ""
        if prefix_file_count > 0:
            image_context_info = f"""

**IMPORTANT - Image Context for OpenCUA:**
- The test model was shown the FIRST {prefix_file_count} images/screenshots (corresponding to the trajectory prefix)
- You are seeing ALL images (both prefix and remaining) to understand the full context
- Images {prefix_file_count + 1} onwards were NOT visible to the test model when making its prediction
- When evaluating, consider: Did the model make reasonable predictions based on ONLY the first {prefix_file_count} images?
"""
        
        prompt = f"""You are evaluating a planning model's next THREE steps prediction.

**Dataset:** {dataset_name}

**User Query:**
{query}

**Trajectory So Far:**
{trajectory_prefix}
{tools_info}
{image_context_info}
{reference_info}

**Predicted Next Three Steps:**
{predicted_steps}

**Note:** The prediction above should contain THREE steps. Please parse and evaluate all three steps.

{error_defs}

{scoring}

**Your Task:**
Evaluate the THREE-STEP SEQUENCE prediction with EXTREME STRICTNESS.

**CRITICAL EVALUATION REQUIREMENTS:**
1. **POTENTIAL ERROR = ERROR:** If ANY step in the sequence *might* fail or has *potential* risks, it is an ERROR.
2. **STRICTEST INTERPRETATION:** If there is any doubt, mark it as an error.
3. **IGNORE INTENT:** Do not evaluate "planning ability" or "intent". Evaluate the strict correctness of the steps.
4. **OPTIMALITY:** Do NOT penalize suboptimal steps. If a step is valid and correct, it is NOT an error, even if it is less efficient than the reference.
5. **SAFETY FIRST:** Any ambiguity is an error.
6. **NO LENIENCY:** Do not be lenient. Be a harsh critic.
7. **SEQUENCE COHERENCE:** Check if the steps logically follow each other. If the sequence is disjointed or a step ignores the result of a previous step, it is an error.

**CRITICAL FOR E2_PREMATURE_CONCLUSION:**
- **CHECK** if ANY step in the sequence has NO tool calls (empty tool_calls array or just text content)
- **BUT** the task clearly requires executing a tool to create/generate/process something
- **EXAMPLE:** Task requires "generate HTML report", reference calls generate_html_report tool, but prediction only says "let me show you the table" or asks "do you want to export?" without actually calling any tool
- This is premature conclusion because the step talks about completing the task without actually doing it
- Text-only responses are acceptable ONLY when the task is genuinely complete or user input is needed

**CRITICAL FOR E4_LOGIC_ERROR:**
- **CHECK** if ANY step's description/thought claims to do data extraction/processing/cleaning
- **BUT** the actual tool call uses already-processed data that doesn't exist in previous steps
- **EXAMPLE:** Step says "extract the last 4 digits of account numbers" but tool parameters already contain extracted data like "XXXXXX0016"
- This is a logic error because the step skips necessary intermediate processing or has illogical reasoning
- The processed data must come from a previous step or be created in THIS step with appropriate tools
- **DEPENDENCY CHECK:** If a step uses data that a previous step is *supposed* to generate, ensure the previous step actually generates it.

**CRITICAL FOR E5_TOOL_USE_ERROR:**
- **STRICT PARAMETER CHECK:** Check if parameters are optimal and correct for ALL steps.
- **VAGUE PARAMETERS:** Any vague or non-specific parameter is an ERROR.
- **POTENTIAL MISUSE:** If the parameter value *might* be wrong or lead to unexpected results, it is an ERROR.
- **SPECIFICATION COMPLIANCE:** Must strictly adhere to tool specifications.

**CRITICAL FOR E6_HALLUCINATION_ERROR:**
- **MUST CHECK:** Verify that every tool name used in ALL predicted steps exists in the "Available Tools" list
- Tool names must match EXACTLY - check spelling, capitalization, and word order
- Even slight variations in tool names (different spelling, word order, or capitalization) should be flagged as hallucination
- The tool name in the predicted step must be identical to one of the tool names listed in Available Tools

Provide:

1. Error classification (True/False for each E1-E6) - BE STRICT, mark True if ANY doubt exists in ANY step
2. Score (0.0, 0.2, 0.4, 0.6, 0.8, or 1.0) - Apply the strict scoring rubric
3. Detailed reasoning explaining your evaluation and why you marked each error
4. Is correct? (YES ONLY if ALL error flags are False, otherwise NO)

**Output Format (JSON):**
```json
{{
    "E1_GOAL_MISALIGNMENT": false,
    "E2_PREMATURE_CONCLUSION": false,
    "E3_CONSTRAINT_VIOLATION": false,
    "E4_LOGIC_ERROR": false,
    "E5_TOOL_USE_ERROR": false,
    "E6_HALLUCINATION_ERROR": false,
    "score": 0.8,
    "reasoning": "The predicted three-step sequence correctly identifies...",
    "is_correct": "YES"
}}
```

Provide ONLY the JSON output, no additional text.
"""
        return prompt
