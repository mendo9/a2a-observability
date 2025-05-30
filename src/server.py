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


class OpenAIAgentWrapper:
    """Wrapper for OpenAI agent to work with A2A framework"""

    def __init__(self):
        # Initialize OpenAI Agent with tools
        self.agent = Agent(
            model="gpt-4o-mini",
            tools=[get_weather, calculate],
            instructions="""You are a helpful AI assistant with access to weather and calculation tools.
            
            You can:
            1. Get weather information for any location using the get_weather tool
            2. Perform basic math calculations using the calculate tool
            
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
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    async def execute(
        self, context: ServerCallContext, event_queue: EventQueue
    ) -> None:
        """Execute the agent with full observability"""

        with tracer.start_as_current_span("a2a_agent_execute") as span:
            try:
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
                        "session_id": getattr(context, "session_id", "unknown"),
                        "input.value": text_content,
                        "session.id": getattr(context, "session_id", "unknown"),
                    }
                )

                if not text_content:
                    text_content = "Hello! How can I help you today?"

                # Invoke the OpenAI agent
                response = await self.agent.invoke(text_content)

                event_queue.enqueue_event(new_agent_text_message(response))

                span.set_attributes(
                    {
                        "response_generated": True,
                        "output.value": response[:200],
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
        id="openai-weather-math-agent",
        name="OpenAI Weather & Math Agent",
        description="An AI agent powered by OpenAI with weather and calculation capabilities",
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
    print("  • get_weather(location) - Get weather for any location")
    print("  • calculate(operation, a, b) - Perform math calculations")
    print("\n💡 Try asking: 'What's the weather in Tokyo?' or 'Calculate 15 * 7'")
    print("\n📈 All OpenAI calls, tool usage, and agent interactions will be")
    print("   automatically traced and visible in Phoenix at http://127.0.0.1:6006")
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
