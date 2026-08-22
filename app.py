import json
import os
import re
import sqlite3
import time
from datetime import datetime

import requests
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic Work Intake & Execution Prototype",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

# Primary model first, then fallback models.
# These are current stable model IDs documented by Google.
GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

DB_NAME = "agent_system.db"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an Agentic Work Intake and Execution Planner.

Your job is to analyze unstructured business work such as:
- emails
- meeting notes
- customer requests
- founder instructions
- internal work requests

Convert the input into a structured, actionable workflow.

You must determine:

1. What the user is asking for.
2. The category of work.
3. The priority.
4. Whether the request can be executed automatically.
5. Whether human approval is required.
6. Whether a tool/capability is required.
7. Whether important information is missing.
8. What action should be taken.
9. What information should be presented to a human reviewer.

Routing rules:

- "execute_automatically"
  Use when the task is safe and sufficiently clear to execute automatically.

- "prepare_for_human_review"
  Use when a human must approve, review, or make a decision before execution.

- "cannot_execute_with_tools"
  Use when the requested task requires a capability/tool that this prototype does not have.

- "requires_clarification"
  Use when important information is missing and execution cannot safely proceed.

Examples:

A simple request to summarize or classify information:
-> execute_automatically

A request to send money, approve a contract, make an important external decision,
or perform a consequential action:
-> prepare_for_human_review

A request requiring an unavailable external system:
-> cannot_execute_with_tools

A request where the destination, amount, recipient, date, or other critical information
is missing:
-> requires_clarification

Never invent missing information.

Return ONLY valid JSON matching the requested schema.
"""


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

AGENT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {
            "type": "STRING"
        },
        "intent": {
            "type": "STRING"
        },
        "category": {
            "type": "STRING"
        },
        "priority": {
            "type": "STRING",
            "enum": [
                "low",
                "medium",
                "high",
                "critical"
            ]
        },
        "route": {
            "type": "STRING",
            "enum": [
                "execute_automatically",
                "prepare_for_human_review",
                "cannot_execute_with_tools",
                "requires_clarification"
            ]
        },
        "action": {
            "type": "STRING"
        },
        "reason": {
            "type": "STRING"
        },
        "missing_information": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        },
        "required_tools": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        },
        "approval_required": {
            "type": "BOOLEAN"
        },
        "execution_result": {
            "type": "STRING"
        }
    },
    "required": [
        "summary",
        "intent",
        "category",
        "priority",
        "route",
        "action",
        "reason",
        "missing_information",
        "required_tools",
        "approval_required",
        "execution_result"
    ]
}


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    """
    Return a SQLite connection.

    timeout helps reduce temporary database locking problems.
    check_same_thread=False makes the connection safer for Streamlit usage.
    """

    return sqlite3.connect(
        DB_NAME,
        timeout=15,
        check_same_thread=False
    )


def init_db():

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_text TEXT NOT NULL,
                summary TEXT,
                intent TEXT,
                category TEXT,
                priority TEXT,
                route TEXT,
                action TEXT,
                reason TEXT,
                missing_information TEXT,
                required_tools TEXT,
                approval_required INTEGER,
                execution_result TEXT,
                model_used TEXT,
                status TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                request_id INTEGER,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                decision TEXT,
                reviewer TEXT,
                comment TEXT
            )
            """
        )

        conn.commit()


# Initialize database when application starts.
init_db()


# ============================================================
# ACTIVITY LOG
# ============================================================

def log_activity(
    event_type,
    message,
    request_id=None
):

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO activity_log (
                created_at,
                request_id,
                event_type,
                message
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                request_id,
                event_type,
                message
            )
        )

        conn.commit()


def get_activity_logs(limit=30):

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                created_at,
                event_type,
                message
            FROM activity_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )

        return cursor.fetchall()


# ============================================================
# API KEY
# ============================================================

def get_gemini_api_key():

    # Streamlit Cloud Secrets
    try:

        if "GEMINI_API_KEY" in st.secrets:

            key = st.secrets["GEMINI_API_KEY"]

            if key:
                return str(key).strip()

    except Exception:
        pass

    # Local environment variable fallback
    key = os.getenv("GEMINI_API_KEY")

    if key:
        return key.strip()

    return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_json_loads(text):

    if not text:
        raise ValueError("Gemini returned an empty response.")

    text = text.strip()

    # Remove markdown JSON fences if the model ever returns them.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return json.loads(text)


def is_transient_error(error):

    error_text = str(error).lower()

    transient_markers = [
        "503",
        "unavailable",
        "service unavailable",
        "high demand",
        "temporarily",
        "429",
        "resource exhausted",
        "rate limit",
        "500",
        "internal server error",
        "502",
        "504",
        "deadline exceeded"
    ]

    return any(
        marker in error_text
        for marker in transient_markers
    )


# ============================================================
# GEMINI AGENT
# ============================================================

def process_intake_with_llm(
    text,
    api_key
):
    """
    Process unstructured work with Gemini.

    Strategy:

    Model 1
       ↓
    transient error?
       ↓
    retry with exponential backoff
       ↓
    still unavailable?
       ↓
    Model 2
       ↓
    Model 3
    """

    if not api_key:

        raise ValueError(
            "Gemini API key is missing. "
            "Configure GEMINI_API_KEY in Streamlit Secrets."
        )

    client = genai.Client(
        api_key=api_key
    )

    last_error = None

    for model_name in GEMINI_MODELS:

        log_activity(
            "MODEL_ATTEMPT",
            f"Trying Gemini model: {model_name}"
        )

        # Maximum of 3 attempts for each model.
        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=AGENT_RESPONSE_SCHEMA,
                    )
                )

                if not response.text:

                    raise RuntimeError(
                        f"{model_name} returned an empty response."
                    )

                result = safe_json_loads(
                    response.text
                )

                # Store model information internally.
                result["_model_used"] = model_name

                log_activity(
                    "MODEL_SUCCESS",
                    f"Successfully processed using {model_name}"
                )

                return result

            except Exception as error:

                last_error = error

                log_activity(
                    "MODEL_ERROR",
                    (
                        f"{model_name} attempt "
                        f"{attempt + 1}/3 failed: {error}"
                    )
                )

                # Only retry temporary server/rate-limit errors.
                if is_transient_error(error):

                    if attempt < 2:

                        wait_seconds = 2 ** attempt

                        # Small additional jitter.
                        time.sleep(
                            wait_seconds
                        )

                        continue

                    # Three attempts failed.
                    # Move to the next model.
                    break

                # Authentication, bad request, invalid schema, etc.
                # should NOT silently move through all models.
                raise

    raise RuntimeError(
        "All Gemini models are temporarily unavailable. "
        f"Last error: {last_error}"
    )


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_available_tools(
    input_text,
    agent_result
):
    """
    Very small safe tool layer for the prototype.

    Currently supports basic URL reachability checking.
    """

    required_tools = agent_result.get(
        "required_tools",
        []
    )

    if not required_tools:
        return "No external tool was required."

    # Find URLs in the original request.
    urls = re.findall(
        r"https?://[^\s<>\"]+",
        input_text
    )

    if urls:

        results = []

        for url in urls[:3]:

            try:

                response = requests.get(
                    url,
                    timeout=10,
                    allow_redirects=True
                )

                results.append(
                    f"{url} -> HTTP {response.status_code}"
                )

            except requests.RequestException as error:

                results.append(
                    f"{url} -> request failed: {error}"
                )

        return "\n".join(results)

    return (
        "The requested tool/capability is not implemented "
        "in this prototype."
    )


# ============================================================
# SAVE REQUEST
# ============================================================

def save_request(
    input_text,
    result,
    status="processed"
):

    model_used = result.get(
        "_model_used",
        "unknown"
    )

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO requests (
                created_at,
                input_text,
                summary,
                intent,
                category,
                priority,
                route,
                action,
                reason,
                missing_information,
                required_tools,
                approval_required,
                execution_result,
                model_used,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                input_text,
                result.get("summary", ""),
                result.get("intent", ""),
                result.get("category", ""),
                result.get("priority", "medium"),
                result.get("route", ""),
                result.get("action", ""),
                result.get("reason", ""),
                json.dumps(
                    result.get(
                        "missing_information",
                        []
                    )
                ),
                json.dumps(
                    result.get(
                        "required_tools",
                        []
                    )
                ),
                1 if result.get(
                    "approval_required",
                    False
                ) else 0,
                result.get(
                    "execution_result",
                    ""
                ),
                model_used,
                status
            )
        )

        request_id = cursor.lastrowid

        conn.commit()

    return request_id


# ============================================================
# APPROVAL QUEUE
# ============================================================

def create_approval(request_id):

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO approvals (
                request_id,
                created_at,
                decision,
                reviewer,
                comment
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                "pending",
                "",
                ""
            )
        )

        conn.commit()


def get_pending_approvals():

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                a.id,
                a.request_id,
                a.created_at,
                r.summary,
                r.priority,
                r.action,
                r.reason,
                r.input_text
            FROM approvals a
            JOIN requests r
                ON a.request_id = r.id
            WHERE a.decision = 'pending'
            ORDER BY a.id DESC
            """
        )

        return cursor.fetchall()


def update_approval(
    approval_id,
    request_id,
    decision,
    reviewer,
    comment
):

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE approvals
            SET
                decision = ?,
                reviewer = ?,
                comment = ?
            WHERE id = ?
            """,
            (
                decision,
                reviewer,
                comment,
                approval_id
            )
        )

        cursor.execute(
            """
            UPDATE requests
            SET status = ?
            WHERE id = ?
            """,
            (
                "approved"
                if decision == "approved"
                else "rejected",
                request_id
            )
        )

        conn.commit()

    log_activity(
        "HUMAN_DECISION",
        (
            f"Request #{request_id} "
            f"was {decision} by {reviewer or 'reviewer'}."
        ),
        request_id
    )


# ============================================================
# DATABASE HISTORY
# ============================================================

def get_request_history(limit=50):

    with get_db_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                created_at,
                summary,
                priority,
                route,
                status,
                model_used
            FROM requests
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )

        return cursor.fetchall()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings & Key")

    api_key = get_gemini_api_key()

    if api_key:

        st.success(
            "Gemini API key loaded from Streamlit Secrets."
        )

    else:

        st.error(
            "Gemini API key is not configured."
        )

        st.info(
            "Add GEMINI_API_KEY in Streamlit Cloud → "
            "Manage app → Settings → Secrets."
        )

    st.divider()

    st.subheader("🤖 AI Configuration")

    st.caption(
        "Primary model:"
    )

    st.code(
        GEMINI_MODELS[0],
        language="text"
    )

    st.caption(
        "Automatic fallback:"
    )

    st.code(
        " → ".join(GEMINI_MODELS),
        language="text"
    )

    st.caption(
        "Google GenAI SDK"
    )

    st.divider()

    st.subheader("📡 Live Activity Trace")

    logs = get_activity_logs(
        limit=20
    )

    if logs:

        for created_at, event_type, message in logs:

            st.markdown(
                f"""
**[{created_at}] {event_type}**

{message}
"""
            )

            st.divider()

    else:

        st.caption(
            "No activity yet."
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚡ Agentic Work Intake & Execution Prototype"
)

st.caption(
    "Altibbe Practical AI Engineering Assignment Submission"
)


# ============================================================
# TABS
# ============================================================

tab_intake, tab_approval, tab_history = st.tabs(
    [
        "📥 Intake & Processing",
        "⏸️ Pending Human Approvals",
        "📊 Database History"
    ]
)


# ============================================================
# TAB 1: INTAKE
# ============================================================

with tab_intake:

    st.header(
        "1. Submit Unstructured Work"
    )

    st.write(
        "Paste an email, meeting note, customer request, "
        "or founder instruction."
    )

    default_text = """
Dear Altibbe Health Team,

Thank you for the update.

I understand the location requirement. I would like to confirm
that I am willing to relocate to Raidurg, Hyderabad, and I will
be able to maintain a sustainable daily commute to T-Hub.

Please let me know the next steps.
""".strip()

    input_text = st.text_area(
        "Paste email, meeting notes, customer request, or founder instruction:",
        value=default_text,
        height=180,
        key="intake_text"
    )

    submit = st.button(
        "🚀 Submit to Agent",
        type="primary",
        use_container_width=False
    )

    if submit:

        if not input_text.strip():

            st.warning(
                "Please enter some work/request text."
            )

        elif not api_key:

            st.error(
                "Gemini API key is not configured."
            )

        else:

            log_activity(
                "INTAKE_RECEIVED",
                f"Text length: {len(input_text)} characters"
            )

            with st.spinner(
                "Agent is analyzing and routing the work..."
            ):

                try:

                    result = process_intake_with_llm(
                        input_text,
                        api_key
                    )

                    route = result.get(
                        "route",
                        "requires_clarification"
                    )

                    # ------------------------------------------------
                    # Automatic execution
                    # ------------------------------------------------

                    if route == "execute_automatically":

                        execution_result = (
                            execute_available_tools(
                                input_text,
                                result
                            )
                        )

                        result["execution_result"] = (
                            execution_result
                        )

                        status = "executed"

                    # ------------------------------------------------
                    # Human approval
                    # ------------------------------------------------

                    elif route == "prepare_for_human_review":

                        result["execution_result"] = (
                            "Execution paused until human approval."
                        )

                        status = "pending_approval"

                    # ------------------------------------------------
                    # Missing tools
                    # ------------------------------------------------

                    elif route == "cannot_execute_with_tools":

                        result["execution_result"] = (
                            "Execution stopped because "
                            "the required capability is unavailable."
                        )

                        status = "blocked"

                    # ------------------------------------------------
                    # Clarification
                    # ------------------------------------------------

                    else:

                        result["execution_result"] = (
                            "Execution paused because "
                            "additional information is required."
                        )

                        status = "needs_clarification"

                    # Save request
                    request_id = save_request(
                        input_text,
                        result,
                        status
                    )

                    # Create approval item if needed
                    if route == "prepare_for_human_review":

                        create_approval(
                            request_id
                        )

                    log_activity(
                        "ROUTED",
                        (
                            f"Request #{request_id} routed to "
                            f"{route}."
                        ),
                        request_id
                    )

                    st.success(
                        "✅ Work processed and routed successfully!"
                    )

                    # ------------------------------------------------
                    # Agent analysis
                    # ------------------------------------------------

                    st.header(
                        "2. Agent Analysis"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Priority",
                            result.get(
                                "priority",
                                "medium"
                            ).upper()
                        )

                    with col2:

                        st.metric(
                            "Route",
                            route.replace(
                                "_",
                                " "
                            ).title()
                        )

                    with col3:

                        st.metric(
                            "Approval",
                            "Required"
                            if result.get(
                                "approval_required",
                                False
                            )
                            else "Not Required"
                        )

                    st.subheader(
                        "📋 Summary"
                    )

                    st.write(
                        result.get(
                            "summary",
                            ""
                        )
                    )

                    st.subheader(
                        "🎯 Intent"
                    )

                    st.write(
                        result.get(
                            "intent",
                            ""
                        )
                    )

                    st.subheader(
                        "🏷️ Category"
                    )

                    st.write(
                        result.get(
                            "category",
                            ""
                        )
                    )

                    st.subheader(
                        "🧭 Routing Decision"
                    )

                    st.info(
                        result.get(
                            "reason",
                            ""
                        )
                    )

                    st.subheader(
                        "⚡ Recommended Action"
                    )

                    st.write(
                        result.get(
                            "action",
                            ""
                        )
                    )

                    missing = result.get(
                        "missing_information",
                        []
                    )

                    if missing:

                        st.subheader(
                            "❓ Missing Information"
                        )

                        for item in missing:

                            st.warning(
                                item
                            )

                    tools = result.get(
                        "required_tools",
                        []
                    )

                    if tools:

                        st.subheader(
                            "🛠️ Required Tools"
                        )

                        for tool in tools:

                            st.write(
                                f"- {tool}"
                            )

                    st.subheader(
                        "▶️ Execution Result"
                    )

                    st.write(
                        result.get(
                            "execution_result",
                            ""
                        )
                    )

                    st.caption(
                        "Model used: "
                        + result.get(
                            "_model_used",
                            "unknown"
                        )
                    )

                    st.subheader(
                        "🔎 Raw Structured Agent Output"
                    )

                    clean_result = {
                        key: value
                        for key, value in result.items()
                        if key != "_model_used"
                    }

                    st.json(
                        clean_result
                    )

                except Exception as error:

                    log_activity(
                        "FAILURE_PATH",
                        f"Error encountered: {error}"
                    )

                    st.error(
                        "Execution Error"
                    )

                    st.code(
                        str(error)
                    )

                    st.info(
                        "If this is a temporary Gemini 503/429 error, "
                        "wait a short time and try again. "
                        "The application automatically retries "
                        "temporary failures and falls back between "
                        "supported Gemini models."
                    )


# ============================================================
# TAB 2: HUMAN APPROVALS
# ============================================================

with tab_approval:

    st.header(
        "⏸️ Pending Human Approvals"
    )

    approvals = get_pending_approvals()

    if not approvals:

        st.success(
            "No pending human approvals."
        )

    else:

        st.write(
            f"{len(approvals)} request(s) waiting for review."
        )

        for approval in approvals:

            (
                approval_id,
                request_id,
                created_at,
                summary,
                priority,
                action,
                reason,
                original_text
            ) = approval

            with st.expander(
                f"Request #{request_id} • {priority.upper()} • {summary}"
            ):

                st.write(
                    "**Created:**",
                    created_at
                )

                st.write(
                    "**Reason for review:**"
                )

                st.info(
                    reason
                )

                st.write(
                    "**Proposed action:**"
                )

                st.write(
                    action
                )

                st.write(
                    "**Original request:**"
                )

                st.text_area(
                    "Original text",
                    value=original_text,
                    height=150,
                    disabled=True,
                    key=f"original_{approval_id}"
                )

                reviewer = st.text_input(
                    "Reviewer name",
                    key=f"reviewer_{approval_id}"
                )

                comment = st.text_area(
                    "Reviewer comment",
                    key=f"comment_{approval_id}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "✅ Approve",
                        key=f"approve_{approval_id}",
                        use_container_width=True
                    ):

                        update_approval(
                            approval_id,
                            request_id,
                            "approved",
                            reviewer,
                            comment
                        )

                        st.success(
                            "Request approved."
                        )

                        st.rerun()

                with col2:

                    if st.button(
                        "❌ Reject",
                        key=f"reject_{approval_id}",
                        use_container_width=True
                    ):

                        update_approval(
                            approval_id,
                            request_id,
                            "rejected",
                            reviewer,
                            comment
                        )

                        st.warning(
                            "Request rejected."
                        )

                        st.rerun()


# ============================================================
# TAB 3: DATABASE HISTORY
# ============================================================

with tab_history:

    st.header(
        "📊 Database History"
    )

    history = get_request_history(
        limit=50
    )

    if not history:

        st.info(
            "No requests have been stored yet."
        )

    else:

        for row in history:

            (
                request_id,
                created_at,
                summary,
                priority,
                route,
                status,
                model_used
            ) = row

            with st.expander(
                f"#{request_id} • {priority.upper()} • {status}"
            ):

                st.write(
                    f"**Created:** {created_at}"
                )

                st.write(
                    f"**Summary:** {summary}"
                )

                st.write(
                    f"**Route:** {route}"
                )

                st.write(
                    f"**Status:** {status}"
                )

                st.write(
                    f"**Model:** {model_used}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Agentic Work Intake & Execution Prototype • "
    "Gemini API • SQLite • Streamlit"
)