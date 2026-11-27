"""
The agent, built with LangChain's create_agent.

create_agent runs the same loop as our hand-written version did: ask the model,
run the tools it asks for, send back the results, and repeat until there's an
answer. Middleware hooks into that loop. Here it keeps the date in the system
prompt current, trims old messages, and caps the number of model calls.

Nothing here knows about Telegram. The bot passes in a conversation id and the
patient's text, and gets back the reply text.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections import defaultdict
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    before_model,
    dynamic_prompt,
)
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from clinic import Clinic
from llm import make_fallback_models, make_model
from tools import make_tools

log = logging.getLogger("clinic-bot.agent")

MAX_MODEL_CALLS = 6  # per patient message: stops a confused model from looping
MAX_HISTORY = 20  # messages kept per chat


def system_prompt(clinic: Clinic) -> str:
    """The instructions the model gets with every call."""
    return f"""You are the AI assistant of {clinic.name}, a dental clinic. You chat with patients on Telegram.
It is now {clinic.now():%A %d %B %Y, %H:%M} at the clinic.

What you help with:
- Questions about services, prices, opening hours, the address and the phone number. Always look these up with the get_clinic_info tool. Never guess or invent prices, services or times.
- Appointments: online booking isn't available yet. Ask the patient to call {clinic.phone}, and mention the opening hours if that helps.

Rules:
- You are an AI assistant, not a person. Say so in your first reply, and whenever someone asks.
- Never give medical advice, a diagnosis or treatment recommendations, even if asked directly. For symptoms, pain or any medical question, say a dentist needs to answer that and give the phone number. If it sounds urgent (severe pain, swelling, heavy bleeding, an injury), tell them to call right away, or to get emergency care if the clinic is closed.
- If someone complains or is unhappy, apologise briefly and give the phone number so a staff member can help.
- Only talk about the clinic. Politely decline anything else.
- Reply in the language the patient writes in.
- Keep replies short and friendly: a few sentences of plain text, no markdown.
"""


def trim(messages: list) -> list | None:
    """Keep about the last MAX_HISTORY messages, starting at a patient message so
    no tool result is left without the request that produced it.
    Returns None when nothing needs trimming."""
    if len(messages) <= MAX_HISTORY:
        return None
    recent = messages[-MAX_HISTORY:]
    for i, message in enumerate(recent):
        if isinstance(message, HumanMessage):
            return recent[i:]
    return None  # one very long turn: leave it alone rather than cut it mid-way


def _name(model: Any) -> str:
    return getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown"


class Agent:
    def __init__(self, clinic: Clinic, model: Any = None, fallbacks: list | None = None):
        self.clinic = clinic
        self.model = model if model is not None else make_model()
        fallbacks = fallbacks if fallbacks is not None else make_fallback_models()
        self.model_name = _name(self.model)
        if fallbacks:
            self.model_name += " (backups: " + ", ".join(_name(m) for m in fallbacks) + ")"

        @dynamic_prompt
        def current_system_prompt(request: ModelRequest) -> str:
            # Rebuilt before every model call, so the date and time are always current.
            return system_prompt(clinic)

        @before_model
        def trim_history(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            # Long chats would otherwise make every call bigger and more expensive.
            kept = trim(state["messages"])
            if kept is None:
                return None
            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}

        middleware = [
            current_system_prompt,
            trim_history,
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ]
        if fallbacks:
            # When the main model fails (a busy free model, an outage), try the
            # backups in order before giving up.
            middleware.append(ModelFallbackMiddleware(*fallbacks))

        self._graph = create_agent(
            model=self.model,
            tools=make_tools(clinic),
            middleware=middleware,
            # Memory per conversation ("thread"). It lives in RAM, so a restart
            # clears it. A database comes with the booking step.
            checkpointer=InMemorySaver(),
        )
        self._threads: dict[str, str] = {}
        self._thread_numbers = itertools.count(1)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def forget(self, conversation_id: str) -> None:
        """Start the conversation over (used by /start): a new thread has no history."""
        self._threads[conversation_id] = f"{conversation_id}-{next(self._thread_numbers)}"

    def _thread_id(self, conversation_id: str) -> str:
        if conversation_id not in self._threads:
            self.forget(conversation_id)
        return self._threads[conversation_id]

    async def reply(self, conversation_id: str, text: str) -> str:
        # One message at a time per chat, so two quick messages can't mix up its history.
        async with self._locks[conversation_id]:
            config = {"configurable": {"thread_id": self._thread_id(conversation_id)}}
            try:
                result = await self._graph.ainvoke(
                    {"messages": [{"role": "user", "content": text}]}, config=config
                )
            except Exception:
                log.exception("Agent run failed")
                # The thread may be left half-finished (say, a tool request with
                # no result), which would break the next call. Start fresh instead.
                self.forget(conversation_id)
                return (
                    "Sorry, I'm having trouble answering right now. "
                    f"Please try again in a minute, or call us: {self.clinic.phone}"
                )
            messages = result["messages"]
            _log_run(messages)
            answer = final_text(messages)
            if not answer:
                log.warning("No final answer (model call limit reached, or an empty reply)")
                self.forget(conversation_id)
                return f"Sorry, I couldn't work that out. Please call us: {self.clinic.phone}"
            return answer


def final_text(messages: list) -> str:
    """The text of the model's last message, if it's a finished answer."""
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or last.tool_calls:
        return ""
    content = last.content
    if isinstance(content, str):
        return content.strip()
    # Some providers send a list of content blocks: keep only the text ones.
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _log_run(messages: list) -> None:
    """Log what happened for the latest patient message: model calls with their token counts, and tool calls."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    for message in messages[start:]:
        if isinstance(message, AIMessage):
            usage = message.usage_metadata or {}
            log.info("LLM call: input_tokens=%s output_tokens=%s", usage.get("input_tokens"), usage.get("output_tokens"))
            for call in message.tool_calls:
                log.info("Tool call: %s %s", call["name"], call["args"])
