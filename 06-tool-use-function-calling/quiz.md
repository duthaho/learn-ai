# Module 06: Tool Use & Function Calling — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: What problem does tool use solve that prompting alone can't?

<details>
<summary>Answer</summary>

LLMs are frozen at training time — they can't access real-time data (current time, live weather, stock prices), perform reliable calculations (they approximate math), or take actions (send emails, write files, query databases). Tool use bridges this gap by letting the LLM request function calls that your code executes. The LLM decides what to do; tools interact with the real world.
</details>

---

### Q2: Why are tool descriptions important — what happens with a vague description?

<details>
<summary>Answer</summary>

The description is what the LLM reads to decide whether and when to use a tool. A vague description like "weather function" doesn't tell the LLM when to choose this tool over others, what parameters to provide, or what results to expect. A good description includes what it does ("Get current weather for a city"), when to use it ("Use when the user asks about weather or temperature"), and what it returns ("Returns temperature and conditions"). Poor descriptions lead to wrong tool selection, missing parameters, and unreliable behavior.
</details>

---

### Q3: Walk through the tool use loop — what are the steps in order?

<details>
<summary>Answer</summary>

1. Send messages + tool definitions to the LLM. 2. Check `finish_reason` — if `"stop"`, the LLM gave a text answer, you're done. If `"tool_calls"`, continue. 3. Extract tool calls from the response (name, arguments, id). 4. Parse arguments with `json.loads()` (they're a JSON string). 5. Execute each function locally. 6. Append the assistant's tool_calls message to the conversation history. 7. Append tool result messages (with matching tool_call_id). 8. Send updated messages back to the LLM. Repeat from step 2 — the LLM may call more tools or give a final answer.
</details>

---

### Q4: What's the difference between tool_choice "auto" and "required"?

<details>
<summary>Answer</summary>

`"auto"` (default): the LLM decides whether to use tools or respond with text directly. For "tell me a joke", it responds without tools. For "what time is it?", it calls get_current_time. `"required"`: the LLM must call at least one tool — it cannot respond with text only. Use this when you know the request needs a tool call. There's also `"none"` (disables tools entirely) and naming a specific tool (forces that exact tool to be called).
</details>

---

### Q5: How do you send a tool execution error back to the LLM?

<details>
<summary>Answer</summary>

Send the error as a normal tool result message: `{"role": "tool", "tool_call_id": "call_abc123", "content": "Error: city 'Xyz' not found. Try a valid city name."}`. Don't raise an exception — send the error as the content string. The LLM can then self-correct: retry with different arguments, ask the user for clarification, or try a different approach. This keeps the tool loop running instead of crashing.
</details>

---

### Q6: What does finish_reason "tool_calls" mean and what should you do next?

<details>
<summary>Answer</summary>

`finish_reason: "tool_calls"` means the LLM wants to call one or more functions instead of giving a text response. The `message.content` is typically `null` and `message.tool_calls` contains the requested calls. You should: (1) append the assistant's message to the conversation history, (2) parse and execute each tool call, (3) append tool result messages with matching `tool_call_id`s, (4) send the updated messages back to the LLM for the next round. Do NOT treat this as a final response.
</details>

---

### Q7: The LLM returns 3 parallel tool_calls. What's the correct way to handle them?

<details>
<summary>Answer</summary>

Execute all three tool calls and send all three results back before making the next API call. Each result must reference the correct `tool_call_id` from the corresponding tool call. The order matters — results must match their tool call IDs. You cannot send partial results (e.g., only 2 of 3) — all tool results must be present for the next LLM call. The standard loop handles this naturally by iterating over all `message.tool_calls`.
</details>

---

### Q8: Why should tool parameters use simple types instead of deeply nested objects?

<details>
<summary>Answer</summary>

Deeply nested schemas increase the chance of the LLM generating malformed arguments — more nesting means more opportunities for structural errors (missing braces, wrong nesting level, misplaced fields). Simple flat schemas like `{"city": "string", "unit": "string"}` are easier for the LLM to fill correctly than `{"location": {"city": "string", "country": "string", "coordinates": {"lat": "number", "lng": "number"}}}`. They also use fewer tokens and are easier to validate. Use enums when options are known, and keep required fields minimal.
</details>
