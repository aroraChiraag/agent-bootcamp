import asyncio
import contextlib
import json
import signal
import sys
from enum import Enum
from typing import Dict, List, Optional
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionToolParam
from pydantic import BaseModel, Field

load_dotenv()

DESCRIPTION_PATH = "Dev/ApplicationDescriptions.csv"
DEPENDENCY_PATH = "Dev/ApplicationDependencies.csv"
MAX_TURNS = 5
AGENT_LLM_NAME = "gemini-2.5-flash"

# ---------- Pydantic schema ----------
class OriginatingSourceAnswer(str, Enum):
    YES = "Yes"
    NO = "No"
    MAYBE = "Maybe"
    NOT_FOUND = "Application Not Found"

class OriginatingSourceResponse(BaseModel):
    answer: OriginatingSourceAnswer = Field(..., description="Classification result.")
    reasoning: str = Field(..., description="Explanation referencing description and dependencies.")
    confidence_score: int = Field(..., ge=0, le=100, description="0 to 100 model confidence for a 'Yes' classification.")

# Pydantic v2: rebuild model to allow proper schema generation
OriginatingSourceResponse.model_rebuild()
# ---------- Load workbook ----------
try:
    apps_df = pd.read_csv(DESCRIPTION_PATH)
    deps_df = pd.read_csv(DEPENDENCY_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to open expected sheets in {DESCRIPTION_PATH} and/or {DEPENDENCY_PATH}: {e}")

apps_df.columns = [c.strip() for c in apps_df.columns]
deps_df.columns = [c.strip() for c in deps_df.columns]
required_apps = ["App ID", "Application Name", "Description", "Business Line(s)", "Type (Contain External feed or not)"]
required_deps = ["From_AppID", "To_AppID", "Reason"]

missing_apps = [c for c in required_apps if c not in apps_df.columns]
missing_deps = [c for c in required_deps if c not in deps_df.columns]

if missing_apps:
    raise RuntimeError(f"Missing columns in 'Application Descriptions' sheet: {missing_apps}")
if missing_deps:
    raise RuntimeError(f"Missing columns in 'Application Dependencies' sheet: {missing_deps}")

apps_index: Dict[str, Dict] = {}
name_to_id: Dict[str, str] = {}

for _, r in apps_df.iterrows():
    aid = str(r["App ID"]).strip()
    name = str(r.get("Application Name", "")).strip()
    apps_index[aid] = {
        "app_id": aid,
        "application_name": name,
        "description": str(r.get("Description", "")),
        "business_lines": str(r.get("Business Line(s)", "")),
        "type_(contain_external_feed_or_not)": str(r.get("Type (Contain External feed or not)", "")),
    }
    name_to_id[name.lower()] = aid

incoming_map: Dict[str, List[Dict]] = {}
outgoing_map: Dict[str, List[Dict]] = {}

for _, r in deps_df.iterrows():
    frm = str(r["From_AppID"]).strip()
    to = str(r["To_AppID"]).strip()
    reason = str(r.get("Reason", ""))
    incoming_map.setdefault(to, []).append({"from_appid": frm, "to_appid": to, "reason": reason})
    outgoing_map.setdefault(frm, []).append({"from_appid": frm, "to_appid": to, "reason": reason})

# ---------- System Instruction ----------
system_message: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        """
        You are an analyst that classifies whether a given application is an originating source for a specific business line.
        You must always return a Tabular Output with fields:
          - App ID
          - Application Name
          - Dependant App 
          - Answer: "Yes", "No", "Maybe", or "Application Not Found"
          - reasoning: Explanation referencing Description and Dependencies and usually 2 or 3 lines long
          - confidence_score: Integer 0 to 100. The score can vary based on description quality, existence of dependency and other factors.
        Follow this structured flow using tool calls:
        1. Extract application details:
           - Use the 'lookup_application' tool to retrieve the application's Description, Business Line(s), and Type (Contain External feed or not).
           - If no matching application exists, return "Application Not Found".
        2. Extract dependencies:
           - Use the 'lookup_dependencies' tool to identify all incoming and outgoing connections for the target application.
           - Incoming dependencies: applications that send data into the target application.
           - Outgoing dependencies: applications that receive data from the target application.
        3. Guidlines for the application otiginates business data and to classify them as “Yes” or "No", think abou the following:
            - USER ENTRY TEST: “Do user ENTER new data through this system?” If No then might not originating.
            - NEW ENTITY TEST: “Does this system CREATE new entities that did not exist before?” Customers, accounts, transactions (not accounting entries) etc. If No then Likely not originating
            - FIRST ENTRY TEST: “Is this the FIRST place this data exists in the bank?” Check description for “receives from”, “ingests from” and also similar reason for dependency. If data comes from elsewhere then it might not be originating sometimes.
            - PRIMARY DATA TEST: “Is this PRIMARY business data or DERIVED data?” Primary examples: trades, deposits, customers, applications etc. Derived examples: accounting entries, aggregations, calculations etc. If DERIVED then might not originating.
            - CONSOLIDATION TEST: “Does this system consolidate or aggregate from multiple sources?” If YES then it is not originating. 
            - If ANY answer suggests “not originating”, classify as No.
            - Also these are just guidlines and there could eb other reason for it being or not being origiantign source. Make sure to weight all the possibilities and provide a confidence score that reflects that thought.
        3. Reasoning Guidlines:
           - Analyze the application's description and dependencies to determine if it originates primary business data.
           - Other Classification rules:
             - "Yes": The application clearly generates new business data (keywords like onboarding, adding new info, lifecycle management, creation, generation of primary data etc.).
             - "No": The application only processes or transforms existing data without adding new business content.
             - "Maybe": Uncertain or ambiguous, or description is unclear. Please classify as Maybe if there is medium or high degree of subjectiveness. 
             - "Application Not Found": No matching application exists.
           - Incorporate dependency context to support reasoning but do not override description evidence. For example:
             - Receiving data from an originating source does not automatically make the target non-originating.
             - Outgoing connections indicate where data flows but do not define origination by themselves. 
        4. Output:
           - Return ONLY the structured Tabular Data.
           - Do not output intermediate reasoning or explanations outside the output.
           - Make the reasoning concise but reference the key elements in Description and Dependency details.
           - Make the font bigger.
        """
    )
}

# ---------- Tools ----------
tools: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_application",
            "description": "Return the application row given App ID or Name.",
            "parameters": {"type": "object", "properties": {"application_id_or_name": {"type": "string"}}, "required": ["application_id_or_name"]},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_dependencies",
            "description": "Return incoming and outgoing dependencies for an App ID.",
            "parameters": {"type": "object", "properties": {"application_id": {"type": "string"}}, "required": ["application_id"]},
            "strict": True,
        },
    },
]

# ---------- Local tool implementations ----------
def _find_application(key: str) -> Optional[Dict]:
    key = str(key).strip()
    if key in apps_index:
        return apps_index[key]
    kl = key.lower()
    if kl in name_to_id:
        return apps_index[name_to_id[kl]]
    for a in apps_index.values():
        if kl in (a.get("application_name") or "").lower():
            return a
    return None

def _get_dependencies(app_id: str) -> Dict:
    inc = incoming_map.get(app_id, [])
    out = outgoing_map.get(app_id, [])
    for dep_list in [inc, out]:
        for d in dep_list:
            d_from = d["from_appid"]
            d_to = d["to_appid"]
            from_app = apps_index.get(d_from, {})
            to_app = apps_index.get(d_to, {})
            d["from_name"] = from_app.get("application_name", d_from)
            d["from_description"] = from_app.get("description", "")
            d["to_name"] = to_app.get("application_name", d_to)
            d["to_description"] = to_app.get("description", "")
    return {"incoming": inc, "outgoing": out}

# ---------- OpenAI client ----------
async_openai_client = AsyncOpenAI()
async def _cleanup_clients() -> None:
    await async_openai_client.close()
def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# ---------- Main ReAct loop ----------
async def react_rag(query: str, history: List[ChatMessage]):
    oai_messages = [system_message, {"role": "user", "content": query}]
    for _ in range(MAX_TURNS):
        completion = await async_openai_client.chat.completions.create(
            model=AGENT_LLM_NAME,
            messages=oai_messages,
            tools=tools,
            reasoning_effort=None
        )
        message = completion.choices[0].message
        oai_messages.append(message)
        history.append(ChatMessage(role="assistant", content=message.content or ""))
        tool_calls = message.tool_calls
        if not tool_calls:
            # Validate structured JSON locally
            parsed_obj = None
            raw_text = message.content or ""
            try:
                parsed = json.loads(raw_text)
                parsed_obj = OriginatingSourceResponse.model_validate(parsed)
            except Exception:
                parsed_obj = OriginatingSourceResponse.model_validate({
                    "answer": OriginatingSourceAnswer.MAYBE.value,
                    "reasoning": f"Model returned unparsable content: {raw_text[:500]}",
                    "confidence_score": 0
                })
            history.append(ChatMessage(role="assistant", content=json.dumps(parsed_obj.model_dump()), metadata={"title": "structured_response"}))
            yield history
            break
        for tool_call in tool_calls:
            fname = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if fname == "lookup_application":
                q = args.get("application_id_or_name")
                res = _find_application(q)
                oai_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(res)})
                history.append(ChatMessage(role="assistant", content=json.dumps(res), metadata={"title": "lookup_application", "q": q}))
            elif fname == "lookup_dependencies":
                aid = args.get("application_id")
                res = _get_dependencies(aid)
                oai_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(res)})
                history.append(ChatMessage(role="assistant", content=json.dumps(res), metadata={"title": "lookup_dependencies", "app_id": aid}))
            else:
                oai_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "null"})
                history.append(ChatMessage(role="assistant", content="null"))
            yield history

# ---------- Gradio UI ----------
demo = gr.ChatInterface(
    react_rag,
    title="Originating Source Agent (structured JSON)",
    type="messages",
    examples=[
        "Is Credit Card Processing an originating source for Retail Banking?",
        "Does PaymentsHub originate data for 'Wealth'?"
    ]
)

# ---------- Run ----------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())