"""
Google A2A Server with OpenAI Agent and Full Observability (Using Official Google A2A SDK)
This implements a complete Google A2A server using the official a2a-sdk with an agent executor pattern
that hosts an OpenAI agent with tracing for both A2A HTTP calls and OpenAI agent LLM/tool calls.
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import re
import random

import nest_asyncio
import logfire
import uvicorn

from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
)
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.apps.starlette_app import A2AStarletteApplication
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.events.event_queue import EventQueue
from a2a.server.context import ServerCallContext
from a2a.server.tasks.task_store import InMemoryTaskStore
from a2a.utils.message import new_agent_text_message

# OpenAI Agents SDK
from agents import Agent, Runner, function_tool

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Phoenix imports
import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

# Apply nest_asyncio for compatibility
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure observability
def setup_observability():
    """Setup OpenTelemetry and Phoenix observability"""

    # Setup Phoenix session
    phoenix_session = px.launch_app()

    # Setup OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Setup Phoenix OTLP exporter
    phoenix_endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"
    )
    phoenix_exporter = OTLPSpanExporter(
        endpoint=phoenix_endpoint,
    )

    # Add Phoenix span processor
    span_processor = BatchSpanProcessor(phoenix_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Also add console exporter for debugging
    console_exporter = ConsoleSpanExporter()
    console_processor = SimpleSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(console_processor)

    # Setup OTLP exporter (optional - for external observability platforms)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers={
                "Authorization": f"Bearer {os.getenv('OTEL_EXPORTER_OTLP_HEADERS', '')}"
            },
        )
        external_span_processor = SimpleSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(external_span_processor)

    # Instrument OpenAI with Phoenix
    OpenAIInstrumentor().instrument()

    # Setup Logfire for OpenAI Agents
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), project_name="a2a-openai-demo")

    return tracer, phoenix_session


# Initialize observability
tracer, phoenix_session = setup_observability()


# Weather function tool for the OpenAI agent
@function_tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        Weather information as a string
    """
    with tracer.start_as_current_span("get_weather_tool") as span:
        span.set_attribute("location", location)
        span.set_attribute("tool.name", "get_weather")

        # Simulate weather API call
        weather_data = {
            "new york": "Sunny, 72°F",
            "london": "Cloudy, 15°C",
            "tokyo": "Rainy, 18°C",
            "paris": "Partly cloudy, 20°C",
        }

        result = weather_data.get(
            location.lower(), f"Weather data not available for {location}"
        )
        span.set_attribute("weather_result", result)
        span.set_attribute("tool.output", result)

        return result


# Math function tool for the OpenAI agent
@function_tool
def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        The result of the calculation
    """
    with tracer.start_as_current_span("calculate_tool") as span:
        span.set_attributes(
            {
                "operation": operation,
                "operand_a": a,
                "operand_b": b,
                "tool.name": "calculate",
            }
        )

        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float("inf"),
        }

        if operation not in operations:
            error_msg = f"Unknown operation: {operation}"
            span.set_attribute("error", error_msg)
            raise ValueError(error_msg)

        result = operations[operation](a, b)
        span.set_attribute("calculation_result", result)
        span.set_attribute("tool.output", str(result))

        return result


# Random testing tool
@function_tool
async def random_testing_tool(test_url: str) -> str:
    """Random validation tool for testing session state transitions.

    Args:
        test_url: A URL to validate (for testing purposes)

    Returns:
        Random success or failure result for testing session flow
    """
    with tracer.start_as_current_span("random_testing_tool") as span:
        span.set_attribute("test_url", test_url)
        span.set_attribute("tool.name", "random_testing_tool")

        # Basic URL format validation (just for testing)
        url_pattern = r"https?://[^\s]+"
        if not re.match(url_pattern, test_url):
            error_msg = f"Invalid URL format: {test_url}"
            span.set_attribute("error", error_msg)
            return error_msg

        span.set_attributes({"validation_type": "random_testing", "test_mode": True})

        # Random validation logic for testing
        # 70% chance of success, 30% chance of failure
        is_valid = random.choice(
            [True, True, True, False, True, True, True, False, True, True]
        )

        if is_valid:
            # Simulate different success messages
            sample_results = [
                "Resource validation successful",
                "Connection test passed",
                "Endpoint is accessible",
                "Service is responding correctly",
                "Test validation completed successfully",
            ]

            result = f"✅ Validation successful: {random.choice(sample_results)}"
            span.set_attributes({"validation_success": True, "random_result": True})
            return result
        else:
            result = f"❌ Validation failed: Unable to validate {test_url}"
            span.set_attributes({"validation_success": False, "random_result": False})
            return result


class OpenAIAgentWrapper:
    """Wrapper for OpenAI agent to work with A2A framework"""

    def __init__(self):
        # Initialize OpenAI Agent with tools
        self.agent = Agent(
            model="gpt-4o-mini",
            tools=[get_weather, calculate, random_testing_tool],
            instructions="""You are a helpful AI assistant with access to weather, calculation, and random testing tools.
            
            You can:
            1. Get weather information for any location using the get_weather tool
            2. Perform basic math calculations using the calculate tool
            3. Validate random testing URLs using the random_testing_tool
            
            Always be helpful and provide clear, accurate responses.""",
        )

        # Initialize agent runner
        self.runner = Runner(self.agent)

    async def invoke(self, message: str) -> str:
        """Invoke the OpenAI agent with a message"""

        with tracer.start_as_current_span("openai_agent_execution") as span:
            span.set_attributes(
                {
                    "agent_model": "gpt-4o-mini",
                    "user_message": message[:200],
                    "span.kind": "llm",
                }
            )

            try:
                # Run the agent - OpenAI instrumentation will automatically trace this
                result = await self.runner.run(message)
                agent_response = str(result)

                span.set_attribute("agent_response", agent_response[:200])
                span.set_attribute("llm.response.model", "gpt-4o-mini")

                return agent_response

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", str(e))
                agent_response = f"I encountered an error: {str(e)}"

                return agent_response


class ObservableAgentExecutor(AgentExecutor):
    """A2A Agent Executor with OpenAI Agent and full observability"""

    def __init__(self):
        super().__init__()
        self.agent = OpenAIAgentWrapper()
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session_id(self, context: ServerCallContext) -> str:
        """Extract or generate session ID from context"""
        # Try to get session ID from context attributes
        session_id = getattr(context, "session_id", None)
        if not session_id:
            # Generate a new session ID based on context info
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        return session_id

    def initialize_session(self, session_id: str, context: ServerCallContext) -> None:
        """Initialize a new session with default state"""
        self.sessions[session_id] = {
            "state": "init",  # init, waiting_for_url, running
            "context_id": getattr(context, "context_id", None),
            "task_id": getattr(context, "task_id", None),
            "created_at": datetime.utcnow().isoformat(),
            "test_url": None,
            "url_validated": False,
            "message_history": [],
        }

    def update_session_state(self, session_id: str, new_state: str, **kwargs) -> None:
        """Update session state and additional data"""
        if session_id in self.sessions:
            self.sessions[session_id]["state"] = new_state
            self.sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
            for key, value in kwargs.items():
                self.sessions[session_id][key] = value

    def is_valid_url(self, text: str) -> bool:
        """Check if the text contains a valid URL"""
        url_pattern = r"https?://[^\s]+"
        return bool(re.search(url_pattern, text))

    async def execute(
        self, context: ServerCallContext, event_queue: EventQueue
    ) -> None:
        """Execute the agent with full observability and session management"""

        with tracer.start_as_current_span("a2a_agent_execute") as span:
            try:
                # Get session ID and initialize if needed
                session_id = self.get_session_id(context)

                if session_id not in self.sessions:
                    self.initialize_session(session_id, context)

                session = self.sessions[session_id]

                # Extract message from context
                message_params = context.params
                message = message_params.get("message", {})

                # Get text content from message parts
                text_content = ""
                for part in message.get("parts", []):
                    if part.get("type") == "text":
                        text_content += part.get("text", "")

                span.set_attributes(
                    {
                        "message_text": text_content[:200],
                        "session_id": session_id,
                        "session_state": session["state"],
                        "input.value": text_content,
                        "session.id": session_id,
                    }
                )

                # Add message to history
                session["message_history"].append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "user",
                        "content": text_content,
                    }
                )

                response = ""

                # Handle different session states
                if session["state"] == "init":
                    # Initial state - ask for URL
                    response = """👋 Hello! I'm your AI assistant with access to weather, calculation, and random testing tools.

To get started, please provide a random testing URL that you'd like me to validate.

Example: https://example.com/test

Once I validate the URL, I'll be able to help you with weather information, calculations, and other tasks!"""

                    self.update_session_state(session_id, "waiting_for_url")

                elif session["state"] == "waiting_for_url":
                    # Waiting for URL
                    if self.is_valid_url(text_content):
                        # Validate the URL
                        validation_result = await random_testing_tool(text_content)

                        if "✅ Validation successful" in validation_result:
                            # URL is valid, update session to running state
                            self.update_session_state(
                                session_id,
                                "running",
                                test_url=text_content,
                                url_validated=True,
                            )

                            response = f"""{validation_result}

🎉 Great! I've validated your URL. Now I'm ready to help you with:
- Weather information for any location
- Mathematical calculations  
- Any other questions you might have

What would you like to know?"""
                        else:
                            # URL validation failed
                            response = f"""{validation_result}

Please provide a valid URL. Make sure the format is:
https://example.com/test

And that the URL exists and is accessible."""
                    else:
                        response = """I need a valid URL to proceed. Please provide a URL in this format:

https://example.com/test

For example: https://example.com/test"""

                elif session["state"] == "running":
                    # URL validated, now use the OpenAI agent normally
                    url_context = f"Working with URL: {session['test_url']}\n\n"
                    full_message = url_context + text_content

                    response = await self.agent.invoke(full_message)

                # Add response to history
                session["message_history"].append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "assistant",
                        "content": response,
                    }
                )

                event_queue.enqueue_event(new_agent_text_message(response))

                span.set_attributes(
                    {
                        "response_generated": True,
                        "output.value": response[:200],
                        "final_session_state": session["state"],
                    }
                )

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", str(e))

                error_response = f"I encountered an error: {str(e)}"
                event_queue.enqueue_event(new_agent_text_message(error_response))


def create_agent_card() -> AgentCard:
    """Create the agent card with capabilities and skills"""

    return AgentCard(
        id="openai-weather-math-random-testing-agent",
        name="OpenAI Weather, Math & Random Testing Agent",
        description="An AI agent powered by OpenAI with weather, calculation, and random testing capabilities",
        version="1.0.0",
        url="http://localhost:8000",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True,
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="random_testing",
                name="Random Testing",
                description="Validate random testing URLs and check if URLs exist",
                examples=["https://example.com/test"],
                tags=["random_testing", "validation", "url"],
            ),
            AgentSkill(
                id="get_weather",
                name="Get Weather",
                description="Get current weather for a location",
                examples=["What's the weather in Tokyo?", "Weather in New York"],
                tags=["weather", "location", "forecast"],
            ),
            AgentSkill(
                id="calculate",
                name="Calculate",
                description="Perform basic mathematical operations",
                examples=["Calculate 15 * 7", "What's 100 / 5?"],
                tags=["math", "calculation", "arithmetic"],
            ),
        ],
    )


async def main():
    """Main function to run the A2A server"""

    print("🚀 Starting A2A Server with OpenAI Agent and Phoenix Observability")
    print("=" * 70)

    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these variables and try again.")
        return

    # Optional observability variables
    observability_vars = ["PHOENIX_COLLECTOR_ENDPOINT", "LOGFIRE_TOKEN"]
    missing_obs = [var for var in observability_vars if not os.getenv(var)]

    if missing_obs:
        print(f"⚠️  Optional observability variables not set: {missing_obs}")
        print("Phoenix will use default local endpoint (http://127.0.0.1:6006)")

    print("\n📊 Observability Features:")
    print("  • OpenTelemetry: ✅ Enabled")
    print(f"  • Phoenix: ✅ Enabled (UI at http://127.0.0.1:6006)")
    print(f"  • OpenAI Instrumentation: ✅ Enabled")
    print(
        f"  • Logfire: {'✅ Enabled' if os.getenv('LOGFIRE_TOKEN') else '⚠️  Disabled'}"
    )

    # Create agent card
    agent_card = create_agent_card()

    # Create agent executor
    agent_executor = ObservableAgentExecutor()

    # Create request handler
    request_handler = RequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    app = A2AStarletteApplication(
        agent_card=agent_card,
        request_handler=request_handler,
    )

    print(f"\n🌐 Server starting on http://localhost:8000")
    print(f"📡 A2A RPC endpoint: http://localhost:8000/")
    print(f"📋 Agent card: http://localhost:8000/.well-known/agent.json")
    print(f"📊 Phoenix UI: http://127.0.0.1:6006")
    print("\n🔧 Available tools:")
    print("  • random_testing(url) - Validate random testing URLs")
    print("  • get_weather(location) - Get weather for any location")
    print("  • calculate(operation, a, b) - Perform math calculations")
    print("\n📋 Session Flow:")
    print("  1️⃣  Initial state: Asks for URL")
    print("  2️⃣  Validation: Checks if URL exists")
    print("  3️⃣  Running: Full agent capabilities unlocked")
    print("\n💡 Example URLs to try:")
    print("  • https://example.com/test")
    print("\n📈 All interactions, OpenAI calls, and tool usage")
    print(
        "   will be automatically traced and visible in Phoenix at http://127.0.0.1:6006"
    )
    print("\nPress Ctrl+C to stop the server")

    try:
        # Run the server
        config = uvicorn.Config(
            app=app.app, host="0.0.0.0", port=8000, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
