import streamlit as st
import sqlite3
import json
import requests
from datetime import datetime
from google import genai
from google.genai import types


# =============================================================================
# CONFIGURATION
# =============================================================================

APP_TITLE = "Agentic Work Intake & Execution Prototype"
DB_NAME = "agent_system.db"
GEMINI_MODEL = "gemini-3.7-flash"


# =============================================================================
# 1. DATABASE & STATE PERSISTENCE
# =============================================================================

def get_db_connection():
    """
    Return a SQLite connection with a timeout.
    check_same_thread=False helps Streamlit's execution model.
    """
    return sqlite3.connect(
        DB_NAME,
        timeout=15,
        check_same_thread=False
    )


def init_db():
    """
    Create all required tables.

    Also performs a small migration for existing databases so that
    older agent_system.db files continue to work.
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # ---------------------------------------------------------------------
        # Requests table
        # ---------------------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT,
                structured_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ---------------------------------------------------------------------
        # Actions table
        # ---------------------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER,
                action_title TEXT,
                action_type TEXT,
                routing_category TEXT,
                reason TEXT,
                status TEXT,
                output TEXT,
                FOREIGN KEY(request_id) REFERENCES requests(id)
            )
            """
        )

        # ---------------------------------------------------------------------
        # Activity logs table
        # ---------------------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                event_type TEXT,
                details TEXT
            )
            """
        )

        # ---------------------------------------------------------------------
        # Database migration
        #
        # We use the existing "output" column to store pending execution
        # parameters, so no additional column is strictly required.
        #
        # This check simply confirms the expected columns exist.
        # ---------------------------------------------------------------------
        cursor.execute("PRAGMA table_info(actions)")
        existing_columns = {
            row[1] for row in cursor.fetchall()
        }

        expected_columns = {
            "request_id",
            "action_title",
            "action_type",
            "routing_category",
            "reason",
            "status",
            "output"
        }

        missing_columns = expected_columns - existing_columns

        if missing_columns:
            # This should normally never happen because CREATE TABLE above
            # handles new installations.
            pass

        conn.commit()


def log_activity(event_type, details, cursor=None):
    """
    Add an activity trace.

    If a cursor is supplied, use the same database transaction.
    Otherwise create a separate connection.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if cursor is not None:
        cursor.execute(
            """
            INSERT INTO activity_logs
            (timestamp, event_type, details)
            VALUES (?, ?, ?)
            """,
            (
                timestamp,
                event_type,
                details
            )
        )
    else:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO activity_logs
                (timestamp, event_type, details)
                VALUES (?, ?, ?)
                """,
                (
                    timestamp,
                    event_type,
                    details
                )
            )
            conn.commit()


# Initialize database when application starts
init_db()


# =============================================================================
# 2. AGENT TOOLS
# =============================================================================

def check_website_status(url: str) -> str:
    """
    Tool 1:
    Check whether a website is reachable.
    """

    if not url:
        return json.dumps({
            "reachable": False,
            "error": "No URL provided."
        })

    url = url.strip()

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    try:
        response = requests.get(
            url,
            timeout=8,
            allow_redirects=True
        )

        return json.dumps({
            "url": url,
            "final_url": response.url,
            "status_code": response.status_code,
            "reachable": 200 <= response.status_code < 400,
            "response_time_seconds": round(
                response.elapsed.total_seconds(),
                2
            )
        })

    except requests.RequestException as e:
        return json.dumps({
            "url": url,
            "reachable": False,
            "error": f"Failed to connect: {str(e)}"
        })

    except Exception as e:
        return json.dumps({
            "url": url,
            "reachable": False,
            "error": str(e)
        })


def create_task_record(
    title: str,
    deadline: str,
    priority: str
) -> str:
    """
    Tool 2:
    Create a task record.

    In this prototype, the tool returns a structured confirmation.
    """

    title = title or "Untitled Task"
    deadline = deadline or "None specified"
    priority = priority or "Medium"

    return json.dumps({
        "status": "CREATED",
        "task_title": title,
        "deadline": deadline,
        "priority": priority,
        "saved_at": datetime.now().isoformat()
    })


def draft_communication(
    recipient: str,
    subject: str,
    body: str
) -> str:
    """
    Tool 3:
    Create a communication draft.
    """

    recipient = recipient.strip() if recipient else ""

    if not recipient:
        recipient_display = "FLAGGED_MISSING_RECIPIENT"
    else:
        recipient_display = recipient

    return json.dumps({
        "status": "DRAFT_SAVED",
        "recipient": recipient_display,
        "subject": subject or "No subject",
        "body": body or "",
        "body_preview": (
            body[:150] + "..."
            if body and len(body) > 150
            else body or ""
        ),
        "saved_at": datetime.now().isoformat()
    })


# =============================================================================
# 3. GEMINI AGENTIC PLANNING
# =============================================================================

SYSTEM_PROMPT = """
You are an Agentic Work Intake Engine.

Your job is to analyze unstructured work instructions such as:

- emails
- meeting notes
- customer requests
- founder instructions
- operational requests

You must convert the input into structured JSON.

You have exactly these three tools:

1. check_website_status
   - Checks whether a website is reachable.
   - Safe to execute automatically.

2. create_task_record
   - Creates a task record.
   - Requires human approval before execution.

3. draft_communication
   - Creates an email/message draft.
   - Requires human approval before execution.

You must route every action into exactly one of these categories:

1. execute_automatically
   Safe actions that can run immediately.

2. prepare_for_human_review
   Actions that should wait for human approval.

3. cannot_execute_with_tools
   Valid requests that cannot be performed by the available tools.

4. requires_clarification
   Requests missing important information.

Important rules:

- Never invent missing information.
- If a recipient is required but missing, use requires_clarification.
- If a URL is explicitly present, use it.
- Do not invent a URL.
- If the user asks to check a website, use check_website_status.
- Creating a task requires human approval.
- Drafting communication requires human approval.
- Explain briefly why an action was routed to its category.

Return ONLY valid JSON matching the supplied schema.
"""


# JSON schema for Gemini structured output
AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "task_title": {
            "type": "string"
        },
        "summary": {
            "type": "string"
        },
        "priority": {
            "type": "string",
            "enum": [
                "High",
                "Medium",
                "Low"
            ]
        },
        "detected_deadline": {
            "type": "string"
        },
        "missing_information": {
            "type": "string"
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action_title": {
                        "type": "string"
                    },
                    "tool_to_use": {
                        "type": "string",
                        "enum": [
                            "check_website_status",
                            "create_task_record",
                            "draft_communication",
                            "none"
                        ]
                    },
                    "routing_category": {
                        "type": "string",
                        "enum": [
                            "execute_automatically",
                            "prepare_for_human_review",
                            "cannot_execute_with_tools",
                            "requires_clarification"
                        ]
                    },
                    "reason": {
                        "type": "string"
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string"
                            },
                            "title": {
                                "type": "string"
                            },
                            "deadline": {
                                "type": "string"
                            },
                            "priority": {
                                "type": "string"
                            },
                            "recipient": {
                                "type": "string"
                            },
                            "subject": {
                                "type": "string"
                            },
                            "body": {
                                "type": "string"
                            }
                        }
                    }
                },
                "required": [
                    "action_title",
                    "tool_to_use",
                    "routing_category",
                    "reason",
                    "parameters"
                ]
            }
        }
    },
    "required": [
        "task_title",
        "summary",
        "priority",
        "detected_deadline",
        "missing_information",
        "actions"
    ]
}


def process_intake_with_llm(
    text: str,
    api_key: str
):
    """
    Send unstructured work to Gemini and receive structured JSON.

    Uses the current Google GenAI SDK and Gemini 3.7 Flash.
    """

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=AGENT_RESPONSE_SCHEMA
        )
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:
        return json.loads(response.text)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini returned invalid JSON: {e}\n\n"
            f"Raw response:\n{response.text}"
        )


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def execute_action(
    tool_name,
    params
):
    """
    Execute one of the three available tools.
    """

    params = params or {}

    if tool_name == "check_website_status":
        return check_website_status(
            params.get("url", "")
        )

    if tool_name == "create_task_record":
        return create_task_record(
            title=params.get("title", ""),
            deadline=params.get("deadline", ""),
            priority=params.get("priority", "")
        )

    if tool_name == "draft_communication":
        return draft_communication(
            recipient=params.get("recipient", ""),
            subject=params.get("subject", ""),
            body=params.get("body", "")
        )

    return json.dumps({
        "status": "NO_TOOL",
        "message": "No executable tool selected."
    })


def safe_json_loads(value, default=None):
    """
    Safely parse JSON stored in SQLite.
    """

    if not value:
        return default if default is not None else {}

    try:
        return json.loads(value)
    except Exception:
        return default if default is not None else {}


def get_api_key():
    """
    Get Gemini API key.

    Priority:
    1. Streamlit Secrets
    2. Manual sidebar input

    This makes the same app work locally and on Streamlit Cloud.
    """

    secret_key = ""

    try:
        secret_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )
    except Exception:
        secret_key = ""

    if secret_key:
        return secret_key

    return st.session_state.get(
        "manual_gemini_api_key",
        ""
    )


# =============================================================================
# 5. STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Work Intake Agent Prototype",
    page_icon="⚡",
    layout="wide"
)


# =============================================================================
# 6. SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("⚙️ Settings & Key")

    # Try Streamlit Secret first
    secret_key_exists = False

    try:
        secret_key_exists = bool(
            st.secrets.get(
                "GEMINI_API_KEY",
                ""
            )
        )
    except Exception:
        secret_key_exists = False

    if secret_key_exists:

        st.success(
            "Gemini API key loaded from Streamlit Secrets."
        )

    else:

        manual_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Used only for this running application."
        )

        st.session_state[
            "manual_gemini_api_key"
        ] = manual_key

    st.markdown("---")

    st.header("🤖 AI Configuration")

    st.caption(
        f"Model: `{GEMINI_MODEL}`"
    )

    st.caption(
        "Google GenAI SDK"
    )

    st.markdown("---")

    st.header("📜 Live Activity Trace")

    try:

        with get_db_connection() as conn:

            logs = conn.execute(
                """
                SELECT
                    timestamp,
                    event_type,
                    details
                FROM activity_logs
                ORDER BY id DESC
                LIMIT 15
                """
            ).fetchall()

    except Exception as e:

        logs = []

        st.error(
            f"Could not load activity logs: {e}"
        )

    if not logs:

        st.caption(
            "No activity yet."
        )

    else:

        for log in logs:

            timestamp = log[0]
            event_type = log[1]
            details = log[2]

            st.caption(
                f"**[{timestamp}] {event_type}**\n"
                f"{details}"
            )


# =============================================================================
# 7. MAIN HEADER
# =============================================================================

st.title(
    "⚡ Agentic Work Intake & Execution Prototype"
)

st.caption(
    "Altibbe Practical AI Engineering Assignment Submission"
)


# =============================================================================
# 8. TABS
# =============================================================================

tab_intake, tab_pending, tab_history = st.tabs(
    [
        "📥 Intake & Processing",
        "⏸️ Pending Human Approvals",
        "📊 Database History"
    ]
)


# =============================================================================
# TAB 1 — INTAKE & PROCESSING
# =============================================================================

with tab_intake:

    st.subheader(
        "1. Submit Unstructured Work"
    )

    user_text = st.text_area(
        "Paste email, meeting notes, customer request, or founder instruction:",
        height=180,
        placeholder=(
            "Example:\n"
            "Please check whether example.com is reachable "
            "and draft an email to the team summarizing the result."
        )
    )

    submit_button = st.button(
        "🚀 Submit to Agent",
        type="primary"
    )

    if submit_button:

        api_key = get_api_key()

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        if not api_key:

            st.error(
                "Please provide your Gemini API key."
            )

        elif not user_text.strip():

            st.warning(
                "Please provide some unstructured work input."
            )

        else:

            log_activity(
                "INTAKE_RECEIVED",
                f"Text snippet: {user_text[:120]}..."
            )

            with st.spinner(
                "🤖 AI Engine Analyzing & Planning..."
            ):

                try:

                    # ---------------------------------------------------------
                    # Gemini planning
                    # ---------------------------------------------------------

                    analysis = process_intake_with_llm(
                        user_text,
                        api_key
                    )

                    # ---------------------------------------------------------
                    # Store request
                    # ---------------------------------------------------------

                    with get_db_connection() as conn:

                        cursor = conn.cursor()

                        cursor.execute(
                            """
                            INSERT INTO requests
                            (raw_text, structured_json)
                            VALUES (?, ?)
                            """,
                            (
                                user_text,
                                json.dumps(
                                    analysis,
                                    ensure_ascii=False
                                )
                            )
                        )

                        request_id = cursor.lastrowid

                        log_activity(
                            "LLM_PARSED",
                            (
                                f"Parsed task: "
                                f"{analysis.get('task_title')}"
                            ),
                            cursor=cursor
                        )

                        # -----------------------------------------------------
                        # Process every planned action
                        # -----------------------------------------------------

                        actions = analysis.get(
                            "actions",
                            []
                        )

                        for action in actions:

                            action_title = action.get(
                                "action_title",
                                "Untitled Action"
                            )

                            tool = action.get(
                                "tool_to_use",
                                "none"
                            )

                            category = action.get(
                                "routing_category",
                                "requires_clarification"
                            )

                            reason = action.get(
                                "reason",
                                ""
                            )

                            params = action.get(
                                "parameters",
                                {}
                            )

                            # =================================================
                            # AUTOMATIC EXECUTION
                            # =================================================

                            if category == "execute_automatically":

                                if tool == "check_website_status":

                                    result = execute_action(
                                        tool,
                                        params
                                    )

                                    status = "COMPLETED"

                                    log_activity(
                                        "AUTO_EXECUTED",
                                        (
                                            f"Tool: {tool} | "
                                            f"Result: {result}"
                                        ),
                                        cursor=cursor
                                    )

                                else:

                                    result = json.dumps({
                                        "status": "NOT_EXECUTED",
                                        "reason": (
                                            "Tool is not configured "
                                            "for automatic execution."
                                        )
                                    })

                                    status = "FLAGGED"

                                    log_activity(
                                        "AUTO_EXECUTION_BLOCKED",
                                        (
                                            f"Tool: {tool} | "
                                            f"Action: {action_title}"
                                        ),
                                        cursor=cursor
                                    )

                            # =================================================
                            # HUMAN APPROVAL
                            # =================================================

                            elif category == "prepare_for_human_review":

                                # Store the actual parameters inside output.
                                # This allows the approval button to execute
                                # the action with the data Gemini extracted.
                                pending_payload = {
                                    "status": "PENDING_APPROVAL",
                                    "tool": tool,
                                    "parameters": params
                                }

                                result = json.dumps(
                                    pending_payload,
                                    ensure_ascii=False
                                )

                                status = "PENDING_APPROVAL"

                                log_activity(
                                    "ROUTED_HITL",
                                    (
                                        f"Action '{action_title}' "
                                        f"paused for human review."
                                    ),
                                    cursor=cursor
                                )

                            # =================================================
                            # FLAGGED
                            # =================================================

                            else:

                                result = json.dumps({
                                    "status": "FLAGGED",
                                    "routing_category": category,
                                    "reason": reason
                                })

                                status = "FLAGGED"

                                log_activity(
                                    "ROUTED_FLAGGED",
                                    (
                                        f"Action: {action_title} "
                                        f"-> {category}"
                                    ),
                                    cursor=cursor
                                )

                            # -------------------------------------------------
                            # Save action
                            # -------------------------------------------------

                            cursor.execute(
                                """
                                INSERT INTO actions
                                (
                                    request_id,
                                    action_title,
                                    action_type,
                                    routing_category,
                                    reason,
                                    status,
                                    output
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    request_id,
                                    action_title,
                                    tool,
                                    category,
                                    reason,
                                    status,
                                    result
                                )
                            )

                        conn.commit()

                    # ---------------------------------------------------------
                    # Success UI
                    # ---------------------------------------------------------

                    st.success(
                        "✅ Work processed and routed successfully!"
                    )

                    st.subheader(
                        "Agent Analysis"
                    )

                    st.json(
                        analysis
                    )

                    # Show summary
                    st.subheader(
                        "Execution Summary"
                    )

                    actions = analysis.get(
                        "actions",
                        []
                    )

                    if actions:

                        for index, action in enumerate(
                            actions,
                            start=1
                        ):

                            category = action.get(
                                "routing_category"
                            )

                            if category == "execute_automatically":
                                icon = "🟢"

                            elif category == "prepare_for_human_review":
                                icon = "🟡"

                            elif category == "requires_clarification":
                                icon = "🟠"

                            else:
                                icon = "🔴"

                            st.write(
                                f"{icon} **{index}. "
                                f"{action.get('action_title')}** — "
                                f"`{category}`"
                            )

                    else:

                        st.info(
                            "No actions were detected."
                        )

                except Exception as e:

                    error_message = str(e)

                    st.error(
                        f"❌ Execution Error: {error_message}"
                    )

                    log_activity(
                        "FAILURE_PATH",
                        (
                            f"Error encountered: "
                            f"{error_message}"
                        )
                    )


# =============================================================================
# TAB 2 — HUMAN-IN-THE-LOOP APPROVAL
# =============================================================================

with tab_pending:

    st.subheader(
        "2. Human-in-the-Loop Approval Queue"
    )

    try:

        with get_db_connection() as conn:

            pending_actions = conn.execute(
                """
                SELECT
                    id,
                    action_title,
                    action_type,
                    reason,
                    output
                FROM actions
                WHERE status = 'PENDING_APPROVAL'
                ORDER BY id DESC
                """
            ).fetchall()

    except Exception as e:

        pending_actions = []

        st.error(
            f"Could not load pending actions: {e}"
        )

    if not pending_actions:

        st.info(
            "No actions currently waiting for approval."
        )

    else:

        st.write(
            f"**{len(pending_actions)} action(s) waiting for approval.**"
        )

        for (
            act_id,
            title,
            tool_type,
            reason,
            output
        ) in pending_actions:

            with st.expander(
                f"🔴 Pending: {title}",
                expanded=True
            ):

                st.write(
                    f"**Action ID:** `{act_id}`"
                )

                st.write(
                    f"**Tool:** `{tool_type}`"
                )

                st.write(
                    f"**Reason for Review:** {reason}"
                )

                # -------------------------------------------------------------
                # Decode the parameters Gemini extracted
                # -------------------------------------------------------------

                pending_data = safe_json_loads(
                    output,
                    {}
                )

                params = pending_data.get(
                    "parameters",
                    {}
                )

                if params:

                    st.write(
                        "**Parameters:**"
                    )

                    st.json(
                        params
                    )

                else:

                    st.warning(
                        "No stored parameters were found."
                    )

                col1, col2 = st.columns(2)

                # =============================================================
                # APPROVE
                # =============================================================

                with col1:

                    approve_button = st.button(
                        "✅ Approve & Execute",
                        key=f"approve_{act_id}",
                        type="primary"
                    )

                    if approve_button:

                        try:

                            # ---------------------------------------------
                            # Execute using the ORIGINAL Gemini parameters
                            # ---------------------------------------------

                            exec_res = execute_action(
                                tool_type,
                                params
                            )

                            # ---------------------------------------------
                            # Update database
                            # ---------------------------------------------

                            with get_db_connection() as conn:

                                conn.execute(
                                    """
                                    UPDATE actions
                                    SET
                                        status = ?,
                                        output = ?
                                    WHERE id = ?
                                    """,
                                    (
                                        "APPROVED_AND_EXECUTED",
                                        exec_res,
                                        act_id
                                    )
                                )

                                conn.commit()

                            log_activity(
                                "HITL_APPROVED",
                                (
                                    f"Action ID {act_id} "
                                    f"approved and executed. "
                                    f"Tool: {tool_type}"
                                )
                            )

                            st.success(
                                "✅ Action approved and executed."
                            )

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Execution failed: {e}"
                            )

                            log_activity(
                                "HITL_EXECUTION_FAILED",
                                (
                                    f"Action ID {act_id}: "
                                    f"{str(e)}"
                                )
                            )

                # =============================================================
                # REJECT
                # =============================================================

                with col2:

                    reject_button = st.button(
                        "❌ Reject",
                        key=f"reject_{act_id}"
                    )

                    if reject_button:

                        try:

                            with get_db_connection() as conn:

                                conn.execute(
                                    """
                                    UPDATE actions
                                    SET
                                        status = ?,
                                        output = ?
                                    WHERE id = ?
                                    """,
                                    (
                                        "REJECTED",
                                        "Rejected by human operator.",
                                        act_id
                                    )
                                )

                                conn.commit()

                            log_activity(
                                "HITL_REJECTED",
                                (
                                    f"Action ID {act_id} "
                                    f"rejected by human."
                                )
                            )

                            st.warning(
                                "Action rejected."
                            )

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Could not reject action: {e}"
                            )


# =============================================================================
# TAB 3 — DATABASE HISTORY
# =============================================================================

with tab_history:

    st.subheader(
        "3. Persistent Storage (SQLite Records)"
    )

    try:

        with get_db_connection() as conn:

            requests_df = conn.execute(
                """
                SELECT
                    id,
                    raw_text,
                    timestamp,
                    structured_json
                FROM requests
                ORDER BY id DESC
                """
            ).fetchall()

    except Exception as e:

        requests_df = []

        st.error(
            f"Could not load database history: {e}"
        )

    if not requests_df:

        st.info(
            "No requests have been processed yet."
        )

    else:

        for req in requests_df:

            request_id = req[0]
            raw_text = req[1]
            timestamp = req[2]
            structured_json = req[3]

            with st.expander(
                f"📌 Request ID #{request_id} — {timestamp}"
            ):

                st.write(
                    "**Raw Input:**"
                )

                st.text(
                    raw_text
                )

                st.write(
                    "**Structured Agent Output:**"
                )

                try:

                    st.json(
                        json.loads(structured_json)
                    )

                except Exception:

                    st.text(
                        structured_json
                    )

                # -------------------------------------------------------------
                # Actions for this request
                # -------------------------------------------------------------

                try:

                    with get_db_connection() as conn:

                        acts = conn.execute(
                            """
                            SELECT
                                id,
                                action_title,
                                action_type,
                                routing_category,
                                reason,
                                status,
                                output
                            FROM actions
                            WHERE request_id = ?
                            ORDER BY id ASC
                            """,
                            (request_id,)
                        ).fetchall()

                except Exception as e:

                    acts = []

                    st.error(
                        f"Could not load actions: {e}"
                    )

                if acts:

                    st.write(
                        "**Actions:**"
                    )

                    for act in acts:

                        (
                            action_id,
                            action_title,
                            action_type,
                            routing_category,
                            reason,
                            status,
                            output
                        ) = act

                        st.markdown(
                            f"### {action_title}"
                        )

                        st.write(
                            f"**Action ID:** `{action_id}`"
                        )

                        st.write(
                            f"**Tool:** `{action_type}`"
                        )

                        st.write(
                            f"**Routing:** `{routing_category}`"
                        )

                        st.write(
                            f"**Status:** `{status}`"
                        )

                        st.write(
                            f"**Reason:** {reason}"
                        )

                        st.write(
                            "**Output:**"
                        )

                        try:

                            st.json(
                                json.loads(output)
                            )

                        except Exception:

                            st.text(
                                output
                            )

                else:

                    st.info(
                        "No actions stored for this request."
                    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.caption(
    "Agentic Work Intake Prototype • "
    f"Powered by {GEMINI_MODEL} • "
    "SQLite Persistence • Human-in-the-Loop"
)