# ⚡ Agentic Work Intake & Execution Prototype

An AI-powered agentic workflow application that converts unstructured work requests into structured execution plans, routes actions based on safety and capability, supports human approval, executes bounded tools, and maintains an activity/history trail.

## 🚀 Live Demo

👉 **(https://byagari-srikanth-agent-prototype-app-p8myzf.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud**.

---

## 🎯 Problem

Business requests often arrive as unstructured emails, meeting notes, customer requests, or instructions.

Manually converting these requests into:

- tasks
- priorities
- required information
- execution steps
- approvals
- tool actions

can be time-consuming and inconsistent.

This project demonstrates how an AI agent can transform unstructured input into a structured and reviewable workflow.

---

## 💡 Solution

The system uses Gemini to:

1. Understand the incoming request
2. Extract the task and summary
3. Identify priority and deadlines
4. Detect missing information
5. Generate an execution plan
6. Route each action based on its risk and capability
7. Automatically execute safe operations
8. Pause actions requiring human approval
9. Store requests and actions in SQLite
10. Maintain a live activity trace

---

## 🧠 Agentic Workflow

```text
                ┌─────────────────────────┐
                │  Unstructured Work      │
                │  Request / Email / Note │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │      Gemini LLM         │
                │ Interpretation + Plan   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   Agentic Router        │
                └────────────┬────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   Execute Automatically   Human Review    Clarification /
          │                  │              Unsupported
          ▼                  ▼
     Tool Execution     Approve / Reject
          │                  │
          └──────────┬───────┘
                     ▼
              SQLite Persistence
                     │
                     ▼
              Activity Trace
