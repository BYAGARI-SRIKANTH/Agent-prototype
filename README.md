# Agentic Work Intake & Execution Prototype

A resilient, single-file AI agent engineering prototype built for **Altibbe Health Pvt. Ltd.** to turn unstructured incoming communications (emails, meeting notes, customer requests) into structured, reviewable, and automated workflows.

---

## 🛠️ Architecture & Workflow

```mermaid
graph TD
    A[Unstructured Intake Text] --> B[LLM Interpretation & Planning - Gemini API]
    B --> C{Agentic Routing Decision}
    C -->|execute_automatically| D[Direct Tool Execution]
    C -->|prepare_for_human_review| E[Pause for HITL Approval Queue]
    C -->|cannot_execute_with_tools| F[Flag Unsupported Capability]
    C -->|requires_clarification| G[Flag Missing Vital Information]
    
    D --> H[(SQLite State Persistence)]
    E -->|Human Click: Approve/Reject| H
    F --> H
    G --> H
    
    H --> I[Live Activity Trace Log UI]