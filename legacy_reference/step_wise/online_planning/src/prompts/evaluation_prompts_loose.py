"""
Evaluation prompts for different datasets (Loose Version).

This module contains dataset-specific evaluation prompts with error classification
frameworks for evaluating agent planning predictions using a looser standard.
"""

import json
from typing import List

from ..utils.dataset_loader import get_base_dataset_type


class EvaluationPromptsLoose:
    """Dataset-specific evaluation prompts with error classification (Loose)."""
    
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
            # FrameThinker format (dialog format with role and content, with think: and action: prefixes)
            format_requirement = """
**CRITICAL FORMAT REQUIREMENT:**
- The predicted step MUST follow the standard dialogue format: `{"role": "assistant", "content": "..."}`
- The `content` field MUST contain both `think:` and `action:` prefixes
- The `think:` section contains the reasoning process
- The `action:` section contains the actual action/tool call
- **EXAMPLE OF CORRECT FORMAT:** `{"role": "assistant", "content": "think: reasoning here\\n\\naction: tool_call_here"}`
- **EXAMPLE OF FORMAT ERROR:** Missing `role` field, missing `think:` or `action:` prefixes
- Format errors should be marked as E3_CONSTRAINT_VIOLATION because the output format is an implicit constraint
- Even if the content itself is correct, missing the required dialogue structure or prefixes is a violation
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
**CRITICAL EVALUATION STANDARDS:**
- Be STRICT and RIGOROUS when detecting errors
- Mark an error as True if there is ANY reasonable doubt about the step's correctness
- Minor deviations, inefficiencies, or suboptimal choices should be marked as errors
- When in doubt between marking an error or not, err on the side of marking it as an error
- A step must be clearly correct and optimal to avoid error flags
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
   - **Definition:** The next step has logical flaws - it requires data/results that haven't been obtained yet, ignores available data from previous steps, or the reasoning flow is illogical.
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
   - **Definition:** The next step misuses a tool's function or passes incorrect parameters **according to the tool's actual specification in the Available Tools list**.
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
   - **Definition:** The next step has logical flaws - missing prerequisites, redundant operations, or illogical reasoning flow.
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
   - **Definition:** The next step misuses tools or passes incorrect parameters **according to the tool's actual specification in the Available Tools list**.
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
   - **Definition:** The next step has logical flaws - uses results not yet calculated, ignores available results, or has illogical reasoning.
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
   - **Definition:** The next step misuses tools (e.g., wrong data types) **according to the tool's actual specification in the Available Tools list**.
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
   - **Definition:** The next step has logical flaws - makes API calls without required prerequisites, has wrong dependencies, or illogical reasoning flow.
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
   - **Definition:** The next step uses incorrect API parameters or data types **according to the API/tool's actual specification in the Available Tools list**.
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
   - **CRITICAL:** Check if the predicted step is missing the required dialogue format structure (role, content) and the required `think:` and `action:` prefixes.
   - **What to check:** Are reasoning constraints respected? Is the required format followed? Does the content include both `think:` and `action:` prefixes?
   - **Examples:**
     * Query says "use only given facts, no external knowledge", next step introduces external assumptions
     * Task specifies "use formal logic notation", next step uses informal natural language
     * Query requires "provide counterexample if exists", next step gives general argument without concrete counterexample
     * Predicted step is `{"name": "...", "reason": "..."}` instead of `{"role": "assistant", "content": "think: ...\\n\\naction: ..."}`
     * Content is missing `think:` or `action:` prefixes
     * Task says "use deductive reasoning only", next step uses inductive generalization
   - **Format violation:** Any predicted step that does not include `role` and `content` fields, or missing `think:` and `action:` prefixes in content, is a constraint violation

**4. E4_LOGIC_ERROR:**
   - **Definition:** The next step has logical flaws - uses conclusions before establishing supporting premises, redundantly re-establishes premises, or has illogical reasoning.
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
   - **Definition:** The next step misuses reasoning tools or methods **according to the tool's actual specification in the Available Tools list**.
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
        
        error_defs = EvaluationPromptsLoose.get_error_definitions(dataset_name)
        scoring = EvaluationPromptsLoose.get_scoring_rubric()
        
        # Build tools information
        # Note: If tools is an empty list, tool information is in the system_prompt
        tools_info = ""
        if tools:  # Only add tools section if tools list is not empty
            tools_info = "\\n**Available Tools:**\\n"
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
                        tools_info += f"- {name}: {desc}\\n"
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
                        tools_info += f"- {name}: {combined_desc}\\n"
                    else:
                        # Standard format: {"tool_name": ..., "description": ...}
                        name = tool.get('tool_name', 'unknown')
                        desc = tool.get('description', 'No description')
                        tools_info += f"- {name}: {desc}\\n"
                else:
                    # For string format (e.g., opencua)
                    tools_info += f"- {tool}\\n"
        else:
            # If tools list is empty, tool information is already in system_prompt
            tools_info = "\\n**Note:** Available tools are defined in the system prompt.\\n"
        
        # Build reference steps section
        reference_info = f"""**Reference Next Step (for context, not ground truth):**
{reference_next_step}"""
        
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
Evaluate the SINGLE NEXT STEP prediction for CORRECTNESS and REASONABLENESS.

**CRITICAL EVALUATION REQUIREMENTS:**
1. Focus on whether the prediction is CORRECT and REASONABLE, not whether it's optimal
2. Mark errors (E1-E6) as True ONLY when there are actual problems (wrong tools, logical flaws, violations)
3. DO NOT penalize for different-but-valid approaches that differ from the reference
4. The reference trajectory is just ONE valid solution - other valid approaches should NOT be marked as errors
5. Ask: "Will this step successfully progress the task?" not "Is this better than the reference?"
6. Different parameter values, search terms, or tool choices are acceptable if they conform to specifications
7. Only penalize actual incorrectness, not suboptimality or inefficiency

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
- **DO NOT** mark E5 as True simply because the prediction uses different parameter names/values than the reference
- **ONLY** mark E5 as True if the tool usage violates the tool's ACTUAL specification from the "Available Tools" section
- **The reference trajectory is just ONE possible valid approach, not the only correct way to use tools**
- **Parameter names and values can differ from reference as long as they conform to tool specifications**
- If both the predicted step and reference step are valid according to tool specifications, E5 should be False
- **ALSO CHECK:** If tool parameters use vague language descriptions instead of concrete, specific values needed to accomplish the task
- **Example of VALID variation:** Tool spec allows "max_results: int", reference uses max_results=10, prediction uses max_results=5 → NOT an error (both valid)
- **Example of VALID variation:** Tool spec allows "mode: str", reference uses mode="strict", prediction uses mode="fuzzy" → NOT an error if both are valid modes
- **Example of INVALID usage:** Task requires statistical data about database (number of primary keys, foreign keys), but predicted step uses vague descriptions like "Defined for each table" or "Link tables for data integrity" instead of concrete numbers. This IS E5 because the parameters are not sufficiently specific to accomplish the task's data requirements.
- **What to actually check:**
  1. Is the tool name correct? (If not, it's E6 not E5)
  2. Are required parameters present?
  3. Are parameter types correct according to tool spec (e.g., int vs string)?
  4. Are parameter values within valid ranges/enums defined by tool spec?
  5. Are parameters specific enough to accomplish the task (not vague placeholders)?
- **DO NOT check:** Whether parameter names match reference, whether parameter values are identical to reference
- Focus on: wrong tool for the task, missing required parameters, invalid parameter types according to tool spec, parameter values outside tool's defined valid range, vague/non-specific parameter values when concrete values are required

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
        
        error_defs = EvaluationPromptsLoose.get_error_definitions(dataset_name)
        scoring = EvaluationPromptsLoose.get_scoring_rubric()
        
        # Build tools information
        tools_info = ""
        if tools:
            tools_info = "\\n**Available Tools:**\\n"
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
                        tools_info += f"- {name}: {desc}\\n"
                    elif 'name' in tool and 'example' in tool:
                        name = tool.get('name', 'unknown')
                        desc = tool.get('description', '')
                        example = tool.get('example', '')
                        combined_desc = f"{desc} Example: {example}" if desc else f"Example: {example}"
                        tools_info += f"- {name}: {combined_desc}\\n"
                    else:
                        name = tool.get('tool_name', 'unknown')
                        desc = tool.get('description', 'No description')
                        tools_info += f"- {name}: {desc}\\n"
                else:
                    tools_info += f"- {tool}\\n"
        else:
            tools_info = "\\n**Note:** Available tools are defined in the system prompt.\\n"
        
        # Build reference steps section
        reference_info = f"""**Reference Next Two Steps (for context, not ground truth):**

{ground_truth_steps}"""
        
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
Evaluate the OVERALL QUALITY of the two predicted steps as a coherent sequence.

**CRITICAL EVALUATION REQUIREMENTS:**
1. Consider both steps together as a complete sequence
2. Evaluate whether the two-step sequence makes meaningful progress toward the goal
3. Check the logical flow: Does Step 2 make sense after Step 1?
4. Focus on whether the predictions are CORRECT and REASONABLE as a whole, not whether they're optimal
5. The reference trajectory is just ONE valid solution - other valid approaches should NOT be marked as errors
6. Mark errors (E1-E6) as True ONLY when there are actual problems in the sequence
7. Give ONE overall score for the entire two-step sequence

**SPECIAL CONSIDERATIONS FOR TWO-STEP EVALUATION:**
- **Coherence**: Do the two steps form a logical sequence?
- **Dependencies**: If Step 2 depends on Step 1's results, is that dependency valid?
- **Goal Progress**: Do these two steps together make meaningful progress toward the goal?
- **Premature Conclusion (E2)**: Does either step conclude prematurely?
- **Tool Usage (E5)**: Are tools used correctly in both steps?
- **Overall Quality**: Rate the quality of the complete two-step sequence, not individual steps

**Output Format (JSON):**
Provide ONE evaluation for the entire two-step sequence:

```json
{{
  "E1_GOAL_MISALIGNMENT": false,
  "E2_PREMATURE_CONCLUSION": false,
  "E3_CONSTRAINT_VIOLATION": false,
  "E4_LOGIC_ERROR": false,
  "E5_TOOL_USE_ERROR": false,
  "E6_HALLUCINATION_ERROR": false,
  "score": 0.8,
  "reasoning": "Overall assessment: The two-step sequence...",
  "is_correct": "YES"
}}
```

**Scoring Guidelines:**
- 1.0: Both steps are excellent and form a perfect sequence
- 0.8: Both steps are good with minor issues
- 0.6: Steps are acceptable but have some problems
- 0.4: Steps have significant issues but show some understanding
- 0.2: Steps have major problems
- 0.0: Steps are completely wrong or irrelevant

**is_correct values:**
- "YES": The two-step sequence is correct and reasonable
- "NO": The two-step sequence has significant errors

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
        
        error_defs = EvaluationPromptsLoose.get_error_definitions(dataset_name)
        scoring = EvaluationPromptsLoose.get_scoring_rubric()
        
        # Build tools information
        tools_info = ""
        if tools:
            tools_info = "\\n**Available Tools:**\\n"
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
                        tools_info += f"- {name}: {desc}\\n"
                    elif 'name' in tool and 'example' in tool:
                        name = tool.get('name', 'unknown')
                        desc = tool.get('description', '')
                        example = tool.get('example', '')
                        combined_desc = f"{desc} Example: {example}" if desc else f"Example: {example}"
                        tools_info += f"- {name}: {combined_desc}\\n"
                    else:
                        name = tool.get('tool_name', 'unknown')
                        desc = tool.get('description', 'No description')
                        tools_info += f"- {name}: {desc}\\n"
                else:
                    tools_info += f"- {tool}\\n"
        else:
            tools_info = "\\n**Note:** Available tools are defined in the system prompt.\\n"
        
        # Build reference steps section
        reference_info = f"""**Reference Next Three Steps (for context, not ground truth):**

{ground_truth_steps}"""
        
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
Evaluate the OVERALL QUALITY of the three predicted steps as a coherent sequence.

**CRITICAL EVALUATION REQUIREMENTS:**
1. Consider all three steps together as a complete sequence
2. Evaluate whether the three-step sequence makes meaningful progress toward the goal
3. Check the logical flow: Does Step 2 make sense after Step 1? Does Step 3 make sense after Step 2?
4. Focus on whether the predictions are CORRECT and REASONABLE as a whole, not whether they're optimal
5. The reference trajectory is just ONE valid solution - other valid approaches should NOT be marked as errors
6. Mark errors (E1-E6) as True ONLY when there are actual problems in the sequence
7. Give ONE overall score for the entire three-step sequence

**SPECIAL CONSIDERATIONS FOR THREE-STEP EVALUATION:**
- **Coherence**: Do the three steps form a logical sequence?
- **Dependencies**: If Step 2 depends on Step 1's results, or Step 3 depends on previous results, are those dependencies valid?
- **Goal Progress**: Do these three steps together make meaningful progress toward the goal?
- **Premature Conclusion (E2)**: Does any step conclude prematurely?
- **Tool Usage (E5)**: Are tools used correctly in all three steps?
- **Overall Quality**: Rate the quality of the complete three-step sequence, not individual steps

**Output Format (JSON):**
Provide ONE evaluation for the entire three-step sequence:

```json
{{
  "E1_GOAL_MISALIGNMENT": false,
  "E2_PREMATURE_CONCLUSION": false,
  "E3_CONSTRAINT_VIOLATION": false,
  "E4_LOGIC_ERROR": false,
  "E5_TOOL_USE_ERROR": false,
  "E6_HALLUCINATION_ERROR": false,
  "score": 0.8,
  "reasoning": "Overall assessment: The three-step sequence...",
  "is_correct": "YES"
}}
```

**Scoring Guidelines:**
- 1.0: All three steps are excellent and form a perfect sequence
- 0.8: All three steps are good with minor issues
- 0.6: Steps are acceptable but have some problems
- 0.4: Steps have significant issues but show some understanding
- 0.2: Steps have major problems
- 0.0: Steps are completely wrong or irrelevant

**is_correct values:**
- "YES": The three-step sequence is correct and reasonable
- "NO": The three-step sequence has significant errors

Provide ONLY the JSON output, no additional text.
"""
        return prompt
