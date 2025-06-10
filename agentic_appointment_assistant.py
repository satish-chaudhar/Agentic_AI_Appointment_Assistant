import os
import re
import logging
from datetime import datetime
from dateutil.parser import parse as dateparse, ParserError
from langgraph import LangGraph, Node, Context
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Please set OPENAI_API_KEY in your .env file")

openai = OpenAI(api_key=OPENAI_API_KEY)

def call_openai(prompt: str) -> str:
    """Call OpenAI for text completion (simple wrapper)."""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def detect_intent(user_input: str) -> str:
    """Simple intent detection based on keywords."""
    appointment_keywords = ["book", "schedule", "meeting", "appointment", "call"]
    if any(word in user_input.lower() for word in appointment_keywords):
        return "appointment"
    # Default to general query
    return "general"

def extract_datetime(text: str):
    """Try to parse date/time from user text."""
    try:
        dt = dateparse(text, fuzzy=True)
        if dt < datetime.now():
            # If date is in the past, ignore it
            return None
        return dt
    except ParserError:
        return None

def extract_mode(text: str):
    modes = {"virtual": ["virtual", "online", "video"], "telephonic": ["phone", "call", "telephonic"]}
    text = text.lower()
    for mode, keywords in modes.items():
        if any(k in text for k in keywords):
            return mode
    return None

# LangGraph Nodes definitions

class StartNode(Node):
    def run(self, ctx: Context):
        ctx["user_input"] = yield "Hello! How can I help you today?"

        intent = detect_intent(ctx["user_input"])
        ctx["intent"] = intent

        if intent == "appointment":
            return "AskDateTime"
        else:
            return "GeneralQuery"

class GeneralQueryNode(Node):
    def run(self, ctx: Context):
        # Use OpenAI to answer general queries
        prompt = f"Answer this general query concisely: {ctx['user_input']}"
        response = call_openai(prompt)
        yield response
        return "Start"

class AskDateTimeNode(Node):
    def run(self, ctx: Context):
        # Try to extract date/time from user input or ask
        if "datetime" not in ctx:
            ctx["datetime_input"] = yield "Please provide your preferred date and time for the appointment."
        else:
            ctx["datetime_input"] = ctx["datetime_input"]

        dt = extract_datetime(ctx["datetime_input"])
        if dt is None:
            ctx["datetime_input"] = yield "Sorry, I didn't catch a valid date/time. Could you please specify again?"
            return "AskDateTime"
        else:
            ctx["datetime"] = dt
            yield f"Got it, you want to schedule for {dt.strftime('%A, %B %d, %Y at %I:%M %p')}."
            return "AskAppointmentMode"

class AskAppointmentModeNode(Node):
    def run(self, ctx: Context):
        if "mode" not in ctx:
            ctx["mode_input"] = yield "Would you prefer the appointment to be Virtual or Telephonic?"
        else:
            ctx["mode_input"] = ctx["mode_input"]

        mode = extract_mode(ctx["mode_input"])
        if mode is None:
            ctx["mode_input"] = yield "Please specify if you want the appointment to be Virtual or Telephonic."
            return "AskAppointmentMode"
        else:
            ctx["mode"] = mode
            yield f"Great! You chose {mode.capitalize()} mode."
            return "CompleteBooking"

class CompleteBookingNode(Node):
    def run(self, ctx: Context):
        dt = ctx.get("datetime")
        mode = ctx.get("mode")
        yield f"Your appointment is booked for {dt.strftime('%A, %B %d, %Y at %I:%M %p')} in {mode.capitalize()} mode. Thank you!"
        return "End"

class FallbackNode(Node):
    def run(self, ctx: Context):
        yield "I'm sorry, I didn't understand that. Can you please rephrase?"
        return "Start"

class EndNode(Node):
    def run(self, ctx: Context):
        yield "Have a great day!"
        return None

# Build LangGraph

def build_graph():
    graph = LangGraph()
    graph.add_node("Start", StartNode())
    graph.add_node("GeneralQuery", GeneralQueryNode())
    graph.add_node("AskDateTime", AskDateTimeNode())
    graph.add_node("AskAppointmentMode", AskAppointmentModeNode())
    graph.add_node("CompleteBooking", CompleteBookingNode())
    graph.add_node("Fallback", FallbackNode())
    graph.add_node("End", EndNode())
    return graph

def main():
    graph = build_graph()
    ctx = Context()
    current_node = "Start"

    while current_node is not None:
        node = graph.get_node(current_node)
        runner = node.run(ctx)
        try:
            prompt = next(runner)
            print("AI:", prompt)
            user_input = input("You: ")
            current_node = runner.send(user_input)
        except StopIteration as e:
            current_node = e.value

if __name__ == "__main__":
    main()
