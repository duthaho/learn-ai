"""
Memory Chatbot — Module 09 Project (Solution)

An interactive CLI chatbot with configurable personas, context window
management (sliding window + summarization), and persistent long-term
memory across sessions.

Run: python solution.py
"""

import os
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion, completion_cost
import tiktoken

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

# Context window configuration
MAX_CONTEXT_TOKENS = 4096   # Conservative default for demo
OUTPUT_RESERVE = 1024       # Reserved for LLM response
SUMMARY_THRESHOLD = 0.75    # Summarize when 75% of budget used

MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"


# ---------------------------------------------------------------------------
# Step 1: Token counting
# ---------------------------------------------------------------------------

def count_tokens(messages: list[dict], model: str = MODEL) -> int:
    """Count tokens in a messages list using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model.split("/")[-1])
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        total += 4  # role + content markers overhead
        total += len(encoding.encode(msg["content"]))
    total += 2  # assistant reply priming
    return total


# ---------------------------------------------------------------------------
# Step 2: Sliding window truncation
# ---------------------------------------------------------------------------

def truncate_messages(messages: list[dict], max_tokens: int,
                      model: str = MODEL) -> list[dict]:
    """Drop oldest message pairs to fit within token budget.

    Always preserves the system prompt at index 0.
    Removes pairs (user + assistant) from the front of history.
    """
    result = list(messages)
    while count_tokens(result, model) > max_tokens and len(result) > 2:
        # Remove the message at index 1 (oldest after system prompt)
        removed = result.pop(1)
        # If the next message is the paired response, remove it too
        if len(result) > 1 and result[1]["role"] != "user":
            result.pop(1)
    return result


# ---------------------------------------------------------------------------
# Step 3: Conversation summarization
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict],
                       model: str = MODEL) -> str:
    """Ask the LLM to summarize a list of messages."""
    conversation_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in messages if m["role"] in ("user", "assistant")
    )

    summary_prompt = (
        "Summarize the following conversation concisely. Include:\n"
        "- Key topics discussed\n"
        "- Important facts the user shared\n"
        "- Decisions made or preferences expressed\n"
        "- Open questions or unresolved topics\n\n"
        "Be concise but preserve all important information.\n\n"
        f"Conversation:\n{conversation_text}"
    )

    response = completion(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Step 4: Context management
# ---------------------------------------------------------------------------

def manage_context(messages: list[dict], max_tokens: int,
                   model: str = MODEL) -> list[dict]:
    """Manage conversation context: summarize if over threshold, truncate if needed.

    Strategy:
    1. Check token usage against budget
    2. If over SUMMARY_THRESHOLD, summarize older messages
    3. Inject summary as a message after the system prompt
    4. If still over budget, truncate remaining messages
    """
    history_budget = max_tokens - OUTPUT_RESERVE
    current_tokens = count_tokens(messages, model)

    if current_tokens <= history_budget:
        return messages

    # Need to manage — summarize older messages
    result = list(messages)

    # Keep system prompt (0) and last 4 messages (recent context)
    if len(result) > 6:
        system_msg = result[0]
        old_messages = result[1:-4]
        recent_messages = result[-4:]

        # Check if there's already a summary message
        has_summary = (len(result) > 1 and
                       result[1].get("role") == "system" and
                       "Previous conversation summary" in result[1].get("content", ""))

        if has_summary:
            # Include existing summary in what we summarize
            old_messages = result[1:-4]
            summary_text = summarize_messages(old_messages, model)
        else:
            summary_text = summarize_messages(old_messages, model)

        summary_msg = {
            "role": "system",
            "content": f"Previous conversation summary:\n{summary_text}",
        }

        result = [system_msg, summary_msg] + recent_messages

        print(f"  [Context managed: summarized {len(old_messages)} messages]")

    # Final safety: truncate if still over budget
    result = truncate_messages(result, history_budget, model)

    return result


# ---------------------------------------------------------------------------
# Step 5: Persona and system prompt
# ---------------------------------------------------------------------------

PERSONAS = {
    "1": {
        "name": "Helpful Assistant",
        "description": "A friendly, concise general-purpose assistant.",
        "style": (
            "Be friendly and concise. Answer questions directly. "
            "If you don't know something, say so honestly."
        ),
    },
    "2": {
        "name": "Coding Tutor",
        "description": "A patient tutor who teaches programming through examples and questions.",
        "style": (
            "Be patient and encouraging. Explain concepts with simple examples. "
            "Ask follow-up questions to check understanding. "
            "Guide the user to discover answers rather than giving them directly."
        ),
    },
    "3": {
        "name": "Creative Writer",
        "description": "An imaginative writing companion who inspires creativity.",
        "style": (
            "Be imaginative and expressive. Use vivid language and metaphors. "
            "Encourage creativity. Offer multiple alternatives when brainstorming. "
            "Celebrate the user's ideas."
        ),
    },
}


def build_system_prompt(persona: dict, memories: dict) -> str:
    """Build system prompt from persona + long-term memories."""
    parts = [f"You are {persona['name']}. {persona['description']}"]
    parts.append(f"\nStyle: {persona['style']}")

    if memories.get("user_facts"):
        facts = "\n".join(f"- {f}" for f in memories["user_facts"])
        parts.append(f"\nWhat you know about the user:\n{facts}")

    if memories.get("conversation_summaries"):
        recent = memories["conversation_summaries"][-3:]
        summaries = "\n".join(f"- {s}" for s in recent)
        parts.append(f"\nRecent conversation history:\n{summaries}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step 6: Long-term memory
# ---------------------------------------------------------------------------

def extract_memories(messages: list[dict],
                     model: str = MODEL) -> list[str]:
    """Extract user facts/preferences from conversation."""
    conversation_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in messages if m["role"] in ("user", "assistant")
    )

    if not conversation_text.strip():
        return []

    extraction_prompt = (
        "Extract key facts about the user from this conversation. "
        "Focus on: name, preferences, interests, background, goals. "
        "Return one fact per line, starting each with '- '. "
        "Only include facts explicitly stated by the user, not assumptions. "
        "If no clear facts are stated, respond with 'None'.\n\n"
        f"Conversation:\n{conversation_text}"
    )

    response = completion(
        model=model,
        messages=[{"role": "user", "content": extraction_prompt}],
        max_tokens=200,
    )

    text = response.choices[0].message.content.strip()
    if text.lower() == "none":
        return []

    facts = []
    for line in text.split("\n"):
        line = line.strip().lstrip("- ").strip()
        if line:
            facts.append(line)
    return facts


def load_memory(memory_file: Path) -> dict:
    """Load persistent memory from JSON file."""
    if memory_file.exists():
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"user_facts": [], "conversation_summaries": []}


def save_memory(memory_data: dict, memory_file: Path) -> None:
    """Save persistent memory to JSON file."""
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Step 7: Chat and main loop
# ---------------------------------------------------------------------------

def chat(messages: list[dict],
         model: str = MODEL) -> tuple[str, dict]:
    """Send messages to LLM and return (response_text, usage_info)."""
    response = completion(model=model, messages=messages)
    message = response.choices[0].message
    usage = response.usage

    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    return message.content, {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost": cost,
    }


def main():
    print("=" * 60)
    print("  Memory Chatbot")
    print("=" * 60)
    print(f"  Model: {MODEL}")

    # Persona selection
    print("\n  Choose a persona:")
    for key, persona in PERSONAS.items():
        print(f"    {key}. {persona['name']}")

    choice = input("  > ").strip()
    if choice not in PERSONAS:
        choice = "1"
    persona = PERSONAS[choice]

    # Load long-term memory
    memories = load_memory(MEMORY_FILE)
    n_facts = len(memories["user_facts"])
    n_summaries = len(memories["conversation_summaries"])

    # Build initial system prompt
    system_prompt = build_system_prompt(persona, memories)
    messages = [{"role": "system", "content": system_prompt}]

    history_budget = MAX_CONTEXT_TOKENS - OUTPUT_RESERVE

    print(f"\n  Persona: {persona['name']}")
    print(f"  Memory: {n_facts} user facts loaded, {n_summaries} conversation summaries loaded")
    print(f"  Context budget: {MAX_CONTEXT_TOKENS} tokens ({OUTPUT_RESERVE} reserved for output)")
    print("  Type /memory, /clear, or /bye\n")

    total_turns = 0
    total_in_tokens = 0
    total_out_tokens = 0
    total_cost = 0.0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.lower() == "/bye":
            break

        if user_input.lower() == "/memory":
            print("\n  Long-term memories:")
            if memories["user_facts"]:
                print("    User facts:")
                for fact in memories["user_facts"]:
                    print(f"      - {fact}")
            else:
                print("    User facts: (none)")
            if memories["conversation_summaries"]:
                print("    Conversation summaries:")
                for summary in memories["conversation_summaries"]:
                    print(f"      - {summary}")
            else:
                print("    Conversation summaries: (none)")
            print()
            continue

        if user_input.lower() == "/clear":
            system_prompt = build_system_prompt(persona, memories)
            messages = [{"role": "system", "content": system_prompt}]
            print("  [Conversation cleared — long-term memory preserved]\n")
            continue

        # --- Regular message ---
        messages.append({"role": "user", "content": user_input})

        # Manage context before sending
        messages = manage_context(messages, MAX_CONTEXT_TOKENS, MODEL)

        # Show token usage
        used = count_tokens(messages, MODEL)
        print(f"\n  [Tokens: {used}/{history_budget} used]")

        # Chat
        response_text, usage_info = chat(messages, MODEL)
        messages.append({"role": "assistant", "content": response_text})

        print(f"\n  Assistant: {response_text}")

        in_tok = usage_info["input_tokens"]
        out_tok = usage_info["output_tokens"]
        cost = usage_info["cost"]
        print(f"\n  Tokens: {in_tok} in + {out_tok} out | Cost: ${cost:.4f}\n")

        total_turns += 1
        total_in_tokens += in_tok
        total_out_tokens += out_tok
        total_cost += cost

    # --- Exit: extract and save memories ---
    if total_turns > 0:
        print("\n  Extracting memories from this conversation...")
        new_facts = extract_memories(messages, MODEL)

        # Add new facts (avoid duplicates)
        existing = set(memories["user_facts"])
        added = []
        for fact in new_facts:
            if fact not in existing:
                memories["user_facts"].append(fact)
                added.append(fact)

        # Add conversation summary
        today = datetime.date.today().isoformat()
        if total_turns >= 2:
            conv_summary = summarize_messages(messages, MODEL)
            memories["conversation_summaries"].append(f"{today}: {conv_summary}")

        save_memory(memories, MEMORY_FILE)

        if added:
            for fact in added:
                print(f'  Saved new fact: "{fact}"')
        else:
            print("  No new facts to save.")

    total_tokens = total_in_tokens + total_out_tokens
    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Turns:        {total_turns}")
    print(f"  Total tokens: {total_tokens} ({total_in_tokens} in + {total_out_tokens} out)")
    print(f"  Total cost:   ${total_cost:.4f}")
    print(f"  Memory file:  {MEMORY_FILE.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
