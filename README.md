# Agentic_AI_Appointment_Assistant
Agentic_AI_Appointment_Assistant

**Agentic Flow Assignment: **AI Appointment Assistant using LangGraph
Objective:
Design and implement an agentic AI assistant using LangGraph that can intelligently interact with users, identify appointment-related intents, extract scheduling preferences, and guide users through selecting the appointment mode.

**Requirements:**
You are to build an agentic flow where the AI assistant can:
Respond to general user queries (e.g., “What services do you offer?”, “Tell me about your team.”).
Detect appointment booking intent in the conversation (e.g., “I’d like to book a meeting”, “Can we schedule a call?”).
Extract and confirm date & time preferences from the user’s message.
Ask for and register the user's preferred appointment mode (Virtual / Telephonic).

Deliverables:
A LangGraph-based implementation (in Python) of the agentic assistant.
A README file explaining:
How to run the project
Assumptions made
Flow design and logic behind key decisions
Test conversations (in a .txt or .md file) that demonstrate:
A general query conversation
An appointment booking flow with all 3 steps handled
An edge case where the assistant has to clarify a missing detail (e.g., no time provided)
Optional: Diagram of the LangGraph node structure (state machine)

**Expectations:**
Use LangGraph to define the conversational state machine.
Use OpenAI or another LLM integration to handle natural language understanding.
Design should be modular and allow for extending the flow (e.g., sending calendar invites, reminders).
Ensure graceful handling of incomplete information (e.g., asking follow-up questions).
Use in-memory state management (persistence is optional but appreciated).

**Bonus Points:**
Support for natural date/time expressions (e.g., “this Friday morning”).
Add a fallback mechanism for unrecognized inputs.
Include a logging mechanism to track the conversation flow state transitions.

**Tech Stack:**
Python 3.10+
LangGraph
LangChain (optional)
OpenAI API (or any other LLM provider)

**How to Run:**
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python agentic_appointment_assistant.py


