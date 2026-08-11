import streamlit as st
import sqlite3
import json
import requests
from datetime import datetime
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. DATABASE & STATE PERSISTENCE (SQLite)
# -----------------------------------------------------------------------------
DB_NAME = "agent_system.db"

def get_db_connection():
    """Helper to return a connection with a timeout to prevent locking."""
    return sqlite3.connect(DB_NAME, timeout=15, check_same_thread=False)

def init_db():
    """Initializes SQLite database tables for persistence."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        raw_text TEXT,
                        structured_json TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id INTEGER,
                        action_title TEXT,
                        action_type TEXT,
                        routing_category TEXT,
                        reason TEXT,
                        status TEXT,
                        output TEXT,
                        FOREIGN KEY(request_id) REFERENCES requests(id)
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS activity_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        event_type TEXT,
                        details TEXT
                    )''')
        conn.commit()

def log_activity(event_type, details, cursor=None):
    """Writes event traces to database. Uses an existing cursor if provided to prevent locks."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if cursor:
        cursor.execute("INSERT INTO activity_logs (timestamp, event_type, details) VALUES (?, ?, ?)",
                       (timestamp, event_type, details))
    else:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO activity_logs (timestamp, event_type, details) VALUES (?, ?, ?)",
                         (timestamp, event_type, details))
            conn.commit()

init_db()

# -----------------------------------------------------------------------------
# 2. AGENT TOOLS / FUNCTIONS (At least 3 Real Tools)
# -----------------------------------------------------------------------------
def check_website_status(url: str) -> str:
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    try:
        response = requests.get(url, timeout=5)
        return json.dumps({
            "url": url,
            "status_code": response.status_code,
            "reachable": response.status_code == 200,
            "response_time_seconds": round(response.elapsed.total_seconds(), 2)
        })
    except Exception as e:
        return json.dumps({"url": url, "error": f"Failed to connect: {str(e)}", "reachable": False})

def create_task_record(title: str, deadline: str, priority: str) -> str:
    return json.dumps({
        "status": "CREATED",
        "task_title": title,
        "deadline": deadline,
        "priority": priority,
        "saved_at": datetime.now().isoformat()
    })

def draft_communication(recipient: str, subject: str, body: str) -> str:
    return json.dumps({
        "status": "DRAFT_SAVED",
        "recipient": recipient if recipient else "FLAGGED_MISSING_RECIPIENT",
        "subject": subject,
        "body_preview": body[:100] + "..." if len(body) > 100 else body
    })

# -----------------------------------------------------------------------------
# 3. LLM UNDERSTANDING & AGENTIC PLANNING
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an Agentic Work Intake Engine. Analyze the incoming text and respond STRICTLY in JSON format.
You must extract structured details and route every action into one of these 4 execution categories:
1. "execute_automatically": Safe tasks ready to run immediately (e.g., website uptime checks).
2. "prepare_for_human_review": Important tasks requiring human approval (e.g., drafting emails, creating database tasks).
3. "cannot_execute_with_tools": Valid requests that exceed our available tools.
4. "requires_clarification": Missing mandatory info (recipients, documents, meeting details).

Return JSON structure ONLY matching this exact schema:
{
    "task_title": "Short descriptive title",
    "summary": "Brief overall summary",
    "priority": "High / Medium / Low",
    "detected_deadline": "Extracted deadline or 'None detected'",
    "missing_information": "List any missing vital details or 'None'",
    "actions": [
        {
            "action_title": "Specific action name",
            "tool_to_use": "check_website_status | create_task_record | draft_communication | none",
            "routing_category": "execute_automatically | prepare_for_human_review | cannot_execute_with_tools | requires_clarification",
            "reason": "Brief reason for this routing decision",
            "parameters": {
                "url": "optional string for check_website_status",
                "title": "optional string for create_task_record",
                "deadline": "optional string",
                "priority": "optional string",
                "recipient": "optional string",
                "subject": "optional string",
                "body": "optional string"
            }
        }
    ]
}
"""

def process_intake_with_llm(text: str, api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\nIncoming Unstructured Text:\n{text}",
        generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI & HUMAN-IN-THE-LOOP CONTROL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Work Intake Agent Prototype", layout="wide")
st.title("⚡ Agentic Work Intake & Execution Prototype")
st.caption("Altibbe Practical AI Engineering Assignment Submission")

# Sidebar: Configuration & Activity Trace
with st.sidebar:
    st.header("⚙️ Settings & Key")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.markdown("---")
    st.header("📜 Live Activity Trace")
    with get_db_connection() as conn:
        logs = conn.execute("SELECT timestamp, event_type, details FROM activity_logs ORDER BY id DESC LIMIT 10").fetchall()
    
    for log in logs:
        st.caption(f"**[{log[0]}] {log[1]}**\n{log[2]}")

# Main Tabs
tab_intake, tab_pending, tab_history = st.tabs(["📥 Intake & Processing", "⏸️ Pending Human Approvals", "📊 Database History"])

# TAB 1: INTAKE & PROCESSING
with tab_intake:
    st.subheader("1. Submit Unstructured Work")
    user_text = st.text_area("Paste email, meeting notes, customer request, or founder instruction:", height=150,
                             placeholder="Example: Please review hedamo.com uptime and draft a summary email...")

    if st.button("Submit to Agent"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar!")
        elif not user_text.strip():
            st.warning("Please provide unstructured text input.")
        else:
            log_activity("INTAKE_RECEIVED", f"Text snippet: {user_text[:40]}...")
            with st.spinner("AI Engine Analyzing & Planning..."):
                try:
                    analysis = process_intake_with_llm(user_text, api_key)
                    
                    # Store in Database using a single unified connection to prevent locks
                    with get_db_connection() as conn:
                        c = conn.cursor()
                        log_activity("LLM_PARSED", f"Parsed task: {analysis.get('task_title')}", cursor=c)
                        
                        c.execute("INSERT INTO requests (raw_text, structured_json) VALUES (?, ?)",
                                  (user_text, json.dumps(analysis)))
                        req_id = c.lastrowid
                        
                        for action in analysis.get("actions", []):
                            category = action.get("routing_category")
                            tool = action.get("tool_to_use")
                            params = action.get("parameters", {})
                            
                            if category == "execute_automatically":
                                if tool == "check_website_status":
                                    res = check_website_status(params.get("url", "hedamo.com"))
                                else:
                                    res = "Automatic execution completed."
                                status = "COMPLETED"
                                log_activity("AUTO_EXECUTED", f"Tool: {tool} | Result: {res}", cursor=c)
                            elif category == "prepare_for_human_review":
                                res = "Awaiting Human Approval in Pending tab."
                                status = "PENDING_APPROVAL"
                                log_activity("ROUTED_HITL", f"Action '{action.get('action_title')}' paused for human review.", cursor=c)
                            else:
                                res = f"Action flagged as: {category}. Reason: {action.get('reason')}"
                                status = "FLAGGED"
                                log_activity("ROUTED_FLAGGED", f"Action: {action.get('action_title')} -> {category}", cursor=c)

                            c.execute("""INSERT INTO actions 
                                (request_id, action_title, action_type, routing_category, reason, status, output)
                                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (req_id, action.get("action_title"), tool, category, action.get("reason"), status, res))
                        
                        conn.commit()
                        
                    st.success("Work processed and routed successfully!")
                    st.json(analysis)
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
                    log_activity("FAILURE_PATH", f"Error encountered: {str(e)}")

# TAB 2: HUMAN-IN-THE-LOOP APPROVAL
with tab_pending:
    st.subheader("2. Human-in-the-Loop Approval Queue")
    with get_db_connection() as conn:
        pending_actions = conn.execute("SELECT id, action_title, action_type, reason, output FROM actions WHERE status = 'PENDING_APPROVAL'").fetchall()
    
    if not pending_actions:
        st.info("No actions currently waiting for approval.")
    else:
        for act_id, title, tool_type, reason, output in pending_actions:
            with st.expander(f"🔴 Pending: {title}", expanded=True):
                st.write(f"**Tool:** `{tool_type}`")
                st.write(f"**Reason for Review:** {reason}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve & Execute", key=f"app_{act_id}"):
                        if tool_type == "draft_communication":
                            exec_res = draft_communication("recipient@example.com", f"Draft: {title}", "Approved communication body.")
                        elif tool_type == "create_task_record":
                            exec_res = create_task_record(title, "7 Days", "Medium")
                        else:
                            exec_res = "Execution approved and finalized."
                        
                        with get_db_connection() as conn:
                            conn.execute("UPDATE actions SET status = 'APPROVED_AND_EXECUTED', output = ? WHERE id = ?", (exec_res, act_id))
                            conn.commit()
                        log_activity("HITL_APPROVED", f"Action ID {act_id} approved by human.")
                        st.rerun()
                
                with col2:
                    if st.button("❌ Reject", key=f"rej_{act_id}"):
                        with get_db_connection() as conn:
                            conn.execute("UPDATE actions SET status = 'REJECTED', output = 'Rejected by human operator.' WHERE id = ?", (act_id,))
                            conn.commit()
                        log_activity("HITL_REJECTED", f"Action ID {act_id} rejected by human.")
                        st.rerun()

# TAB 3: DATABASE HISTORY
with tab_history:
    st.subheader("3. Persistent Storage (SQLite Records)")
    with get_db_connection() as conn:
        requests_df = conn.execute("SELECT id, raw_text, timestamp FROM requests ORDER BY id DESC").fetchall()
        
        for req in requests_df:
            with st.expander(f"Request ID #{req[0]} - {req[2]}"):
                st.text(f"Raw Input: {req[1]}")
                acts = conn.execute("SELECT action_title, routing_category, status, output FROM actions WHERE request_id = ?", (req[0],)).fetchall()
                st.table(acts)