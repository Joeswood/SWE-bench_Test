import os
import subprocess
import tempfile
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if exists
except ImportError:
    pass  # dotenv is optional

try:
    # LangGraph is expected to be available per user requirement
    from langgraph.graph import StateGraph, END
except Exception as e:  # pragma: no cover
    raise ImportError(
        "langgraph is required for langgraph_patch_flow. Please install `langgraph`."
    ) from e

try:
    from langchain_openai import ChatOpenAI
except Exception as e:  # pragma: no cover
    raise ImportError(
        "langchain-openai is required for langgraph_patch_flow. Please install `langchain-openai`."
    ) from e

# Optional LangSmith / LangChain tracing imports (best-effort, no hard dependency)
try:  # pragma: no cover
    from langsmith import traceable as _traceable
except Exception:  # pragma: no cover
    def _traceable(*args: Any, **kwargs: Any):
        def _decorator(fn):
            return fn
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator

try:  # pragma: no cover
    from langchain_core.tracers import tracing_v2_enabled as _tracing_v2_enabled
except Exception:  # pragma: no cover
    _tracing_v2_enabled = None

# Note: ChatOpenAI natively supports LangSmith tracing, no wrapping needed


# Model token pricing (duplicated to avoid circular import). Keep in sync with run_api.py
_MODEL_COST_PER_INPUT: Dict[str, float] = {
    "claude-instant-1": 0.00000163,
    "claude-2": 0.00001102,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-sonnet-20240229": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
    "gpt-3.5-turbo-16k-0613": 0.0000015,
    "gpt-3.5-turbo-0613": 0.0000015,
    "gpt-3.5-turbo-1106": 0.000001,
    "gpt-35-turbo-0613": 0.0000015,
    "gpt-35-turbo": 0.0000015,
    "gpt-4-0613": 0.00003,
    "gpt-4-32k-0613": 0.00006,
    "gpt-4-32k": 0.00006,
    "gpt-4-1106-preview": 0.00001,
    "gpt-4-0125-preview": 0.00001,
    "openai/gpt-5": 0.00000125,
    "x-ai/grok-code-fast-1": 0.0000002,
}

_MODEL_COST_PER_OUTPUT: Dict[str, float] = {
    "claude-instant-1": 0.00000551,
    "claude-2": 0.00003268,
    "claude-3-opus-20240229": 0.000075,
    "claude-3-sonnet-20240229": 0.000015,
    "claude-3-haiku-20240307": 0.00000125,
    "gpt-3.5-turbo-16k-0613": 0.000002,
    "gpt-3.5-turbo-16k": 0.000002,
    "gpt-3.5-turbo-1106": 0.000002,
    "gpt-35-turbo-0613": 0.000002,
    "gpt-35-turbo": 0.000002,
    "gpt-4-0613": 0.00006,
    "gpt-4-32k-0613": 0.00012,
    "gpt-4-32k": 0.00012,
    "gpt-4-1106-preview": 0.00003,
    "gpt-4-0125-preview": 0.00003,
    "openai/gpt-5": 0.00001,
    "x-ai/grok-code-fast-1": 0.0000015,
}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return (
        _MODEL_COST_PER_INPUT.get(model, 0.0) * input_tokens
        + _MODEL_COST_PER_OUTPUT.get(model, 0.0) * output_tokens
    )


class _Usage:
    """Usage object to match OpenAI response structure."""
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Message:
    """Message object to match OpenAI response structure."""
    def __init__(self, content: str):
        self.content = content


class _Choice:
    """Choice object to match OpenAI response structure."""
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    """Minimal response shim to match OpenAI ChatCompletion response shape used by run_api.
    
    This class mimics the OpenAI ChatCompletion response structure so that code expecting
    response.choices[0].message.content will work correctly.
    """

    def __init__(self, model: str, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.model = model
        self.usage = _Usage(prompt_tokens, completion_tokens)
        self.choices = [_Choice(content)]


class PatchState(TypedDict, total=False):
    inputs: str
    model: str
    temperature: float
    top_p: float
    feedback: str
    patch: str
    status: str  # "retry" | "ok"
    used_prompt_tokens: int
    used_completion_tokens: int
    total_cost: float
    attempts: int  # Number of generation attempts
    max_retries: int  # Maximum number of retries allowed
    message_history: List[Dict[str, str]]  # Conversation history for multi-turn dialogue


def _build_messages(
    inputs: str, 
    feedback: Optional[str], 
    message_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build messages for LLM call, preserving conversation history for multi-turn dialogue.
    
    Uses the same format as run_api.py:call_chat:
    - First line of inputs: system message
    - Rest of inputs: user message
    
    Args:
        inputs: Original input (first line = system, rest = user)
        feedback: Feedback from reviewer (if retry)
        message_history: Previous conversation history (if retry)
    
    Returns:
        List of messages in OpenAI format
    """
    # If we have message history (retry), extend it with feedback
    if message_history and feedback:
        # message_history already contains: system, user, assistant (previous patch)
        # We append a new user message with feedback
        messages = list(message_history)  # Copy to avoid mutation
        feedback_message = (
            f"[Reviewer Feedback]\n{feedback}\n\n"
            f"Please fix the issues and output ONLY a valid unified diff. "
            f"Remember:\n"
            f"- Every line in a hunk must start with ' ', '+', or '-'\n"
            f"- Include proper headers: 'diff --git', '---', '+++', '@@'\n"
            f"- No code fences (```) or explanatory text\n"
        )
        messages.append({"role": "user", "content": feedback_message})
        return messages
    
    # First attempt: build initial messages
    # Use the same split logic as run_api.py:call_chat (line 160-161)
    system_messages = inputs.split("\n", 1)[0]
    user_message = inputs.split("\n", 1)[1]
    
    # Add strict output constraints to user message
    augmented_user = (
        f"{user_message}\n\nRequirements:\n"
        f"- Output only a unified diff that can be applied by `git apply -p0` or `git apply`.\n"
        f"- Must include standard headers: 'diff --git', '---', '+++', '@@'.\n"
        f"- Do not use Markdown code fences (e.g., ``` or ~~~).\n"
        f"- Every line inside a hunk must start with ' ' (context), '+' (addition), or '-' (deletion).\n"
    )
    
    # Build messages in the same format as run_api.py:call_chat (line 169-172)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_messages},
        {"role": "user", "content": augmented_user},
    ]
    return messages


def _call_openai(model: str, messages: List[Dict[str, str]], temperature: float, top_p: float, **model_args: Any):
    """Call OpenAI API using LangChain's ChatOpenAI for better integration."""
    # ChatOpenAI automatically reads from environment variables:
    # - OPENAI_API_KEY or OPENROUTER_API_KEY (via OPENAI_API_KEY fallback)
    # - OPENAI_BASE_URL (for OpenRouter or custom endpoints)
    
    # Prepare API key and base URL
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = None
    if os.environ.get("OPENROUTER_API_KEY"):
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Create ChatOpenAI instance
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"top_p": top_p, **model_args},
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    
    # Convert messages to LangChain format
    from langchain_core.messages import SystemMessage, HumanMessage
    
    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
    
    # Invoke with usage tracking
    resp = llm.invoke(lc_messages)
    
    # Extract content and token usage
    content = resp.content
    # LangChain's response_metadata contains usage info
    usage = resp.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    return content, prompt_tokens, completion_tokens


def _looks_like_unified_diff(text: str) -> bool:
    if not text:
        return False
    if "```" in text or "~~~" in text:
        return False
    # Must have at least one hunk
    if "@@" not in text:
        return False
    # Prefer full headers when present
    has_diff_header = any(line.startswith("diff --git ") for line in text.splitlines())
    has_file_headers = ("--- " in text) and ("+++ " in text)
    if not (has_diff_header or has_file_headers):
        return False
    # At least one add/remove line
    has_changes = any(line.startswith("+") or line.startswith("-") for line in text.splitlines())
    if not has_changes:
        return False
    return True


@_traceable(name="diff_generator")
def diff_generator(state: PatchState) -> PatchState:
    # Increment attempt counter
    attempts = state.get("attempts", 0) + 1
    max_retries = state.get("max_retries", 10)
    
    # Log generation attempt
    if attempts == 1:
        print(f"\n [LangGraph] Generating patch (attempt 1/{max_retries + 1})...")
    else:
        feedback = state.get("feedback", "")
        print(f"\n [LangGraph] Retrying patch generation (attempt {attempts}/{max_retries + 1})")
        if feedback:
            # Show first few lines of feedback
            feedback_preview = feedback.split('\n')[0][:150]
            print(f"  Feedback: {feedback_preview}{'...' if len(feedback) > 150 else ''}")
    
    # Build messages with conversation history for multi-turn dialogue
    message_history = state.get("message_history")
    messages = _build_messages(
        state["inputs"], 
        state.get("feedback"),
        message_history
    )
    
    content, ptok, ctok = _call_openai(
        model=state["model"],
        messages=messages,
        temperature=state.get("temperature", 0.0),
        top_p=state.get("top_p", 0.1), #  experience from community
    )
    used_prompt = state.get("used_prompt_tokens", 0) + ptok
    used_completion = state.get("used_completion_tokens", 0) + ctok
    total_cost = state.get("total_cost", 0.0) + _calc_cost(state["model"], ptok, ctok)
    
    # Log token usage
    print(f"  Tokens: +{ptok} prompt, +{ctok} completion (total: {used_prompt + used_completion})")
    
    # Update message history: append assistant's response
    # This preserves the conversation for the next retry
    updated_history = list(messages)  # Copy current messages
    updated_history.append({"role": "assistant", "content": content})
    
    return {
        "patch": content,
        "used_prompt_tokens": used_prompt,
        "used_completion_tokens": used_completion,
        "total_cost": total_cost,
        "attempts": attempts,
        "message_history": updated_history,
    }


def _fuzzy_patch_check(patch: str) -> Tuple[bool, str]:
    """Use `patch --dry-run --fuzz=5` for more lenient patch validation.
    
    This is more forgiving than git apply and can:
    - Ignore extra text before/after the patch
    - Handle minor context mismatches (up to 5 lines)
    - Better tolerate whitespace differences
    """
    if not patch.strip():
        return False, "Patch is empty"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".diff") as tf:
        tf.write(patch)
        tf.flush()
        patch_path = tf.name
    try:
        # Try with fuzz=5 for maximum tolerance, then with different -p levels
        for p_level in ("1", "0", ""):
            p_arg = [f"-p{p_level}"] if p_level else []
            cmd = ["patch", "--batch", "--dry-run", "--fuzz=5", *p_arg, "-i", patch_path]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=".")
            if proc.returncode == 0:
                return True, ""
            last_err = proc.stderr.strip() or proc.stdout.strip()
        return False, f"patch dry-run failed: {last_err}"
    finally:
        try:
            os.unlink(patch_path)
        except Exception:
            pass


@_traceable(name="patch_reviewer")
def patch_reviewer(state: PatchState) -> PatchState:
    patch = state.get("patch", "")
    attempts = state.get("attempts", 0)
    max_retries = state.get("max_retries", 10)
    
    print(f"\n🔍 [LangGraph] Reviewing patch (attempt {attempts}/{max_retries + 1})...")

    # Step 1: heuristic check
    print(f"   Step 1/2: Heuristic check...", end=" ")
    ok1 = _looks_like_unified_diff(patch)
    print("✅ PASSED" if ok1 else "❌ FAILED")

    # Step 2: fuzzy patch check
    print(f"   Step 2/2: Fuzzy patch check...", end=" ")
    ok2, err_msg = _fuzzy_patch_check(patch)
    if ok2:
        print("✅ PASSED")
    else:
        print("❌ FAILED")
        if err_msg:
            print(f"      └─ {err_msg[:100]}...")

    # Return result
    if ok1 and ok2:
        print(f"\n✅ [LangGraph] Patch validation SUCCESSFUL! 🎉")
        return {"status": "ok", "feedback": ""}
    
    if attempts >= max_retries:
        print(f"\n⚠️  [LangGraph] Max retries ({max_retries}) reached. Accepting current patch.")
        return {"status": "ok", "feedback": ""}
    
    # Retry with simple feedback
    print(f"\n❌ [LangGraph] Patch validation FAILED. Retrying...")
    feedback = "Patch validation failed. Please generate a valid unified diff format patch."
    return {"status": "retry", "feedback": feedback}


def _should_retry(state: PatchState) -> str:
    return "diff_generator" if state.get("status") == "retry" else END


def _build_graph():
    """Build LangGraph with automatic retry logic.
    
    Flow:
    1. diff_generator generates a patch
    2. patch_reviewer validates the patch
    3. If valid (status="ok"): END
    4. If invalid (status="retry"): go back to diff_generator with feedback
    """
    g = StateGraph(PatchState)
    g.add_node("diff_generator", diff_generator)
    g.add_node("patch_reviewer", patch_reviewer)
    g.set_entry_point("diff_generator")
    g.add_edge("diff_generator", "patch_reviewer")
    # Use conditional edge to implement retry logic within the graph
    g.add_conditional_edges(
        "patch_reviewer",
        _should_retry,
        {
            "diff_generator": "diff_generator",  # retry: feedback loop
            END: END,  # ok: terminate
        }
    )
    return g.compile()


def call_chat_via_langgraph(
    model_name_or_path: str,
    inputs: str,
    use_azure: bool,
    temperature: float,
    top_p: float,
    max_retries: Optional[int] = None,
    **model_args: Any,
):
    """
    Execute a two-node LangGraph to generate a patch and validate it. Returns a response-like object and cost
    compatible with run_api.openai_inference expectations.
    """
    if max_retries is None:
        try:
            max_retries = int(os.environ.get("LANGGRAPH_MAX_RETRIES", "10"))
        except Exception:
            max_retries = 2 # default to 2 retries

    app = _build_graph()

    state: PatchState = {
        "inputs": inputs,
        "model": model_name_or_path,
        "temperature": temperature,
        "top_p": top_p,
        "feedback": "",
        "used_prompt_tokens": 0,
        "used_completion_tokens": 0,
        "total_cost": 0.0,
        "attempts": 0,
        "max_retries": max_retries,
        "message_history": [],  # Will be populated after first generation
    }

    # Enable LangSmith tracing if requested via env and available
    # Note: ChatOpenAI natively supports LangSmith, no manual wrapping needed
    enable_tracing = str(os.environ.get("LANGCHAIN_TRACING_V2", "")).strip().lower() in {"1", "true", "yes"}
    project_name = os.environ.get("LANGCHAIN_PROJECT")

    # Log workflow start
    print(f"\n{'='*70}")
    print(f"  [LangGraph] Starting patch generation workflow")
    print(f"   Model: {model_name_or_path}")
    print(f"   Max retries: {max_retries}")
    print(f"   Validation: Automatic fuzzy (patch --fuzz=5)")
    if enable_tracing:
        print(f"   LangSmith tracing: enabled (project: {project_name or 'default'})")
    print(f"{'='*70}")

    # Invoke the graph - retry logic is handled internally via conditional edges
    # The max_retries limit is enforced in patch_reviewer node
    if enable_tracing and _tracing_v2_enabled is not None:
        try:  # pragma: no cover
            with _tracing_v2_enabled(project_name=project_name):
                state = app.invoke(state)
        except Exception:
            state = app.invoke(state)
    else:
        state = app.invoke(state)

    final_patch = state.get("patch", "")
    prompt_tokens = int(state.get("used_prompt_tokens", 0))
    completion_tokens = int(state.get("used_completion_tokens", 0))
    total_cost = float(state.get("total_cost", 0.0))
    final_attempts = int(state.get("attempts", 0))
    
    # Log workflow completion
    print(f"\n{'='*70}")
    print(f"🏁 [LangGraph] Workflow completed!")
    print(f"   Total attempts: {final_attempts}")
    print(f"   Total tokens: {prompt_tokens + completion_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Patch size: {len(final_patch)} characters")
    print(f"{'='*70}\n")

    response_like = _Response(
        model=model_name_or_path,
        content=final_patch,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return response_like, total_cost


