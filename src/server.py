"""
Google A2A Server with OpenAI Agent and Full Observability (Using Official Google A2A SDK)
This implements a complete Google A2A server using the official a2a-sdk with an agent executor pattern
that hosts an OpenAI agent with tracing for both A2A HTTP calls and OpenAI agent LLM/tool calls.
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any
import logging
import re
import random

import nest_asyncio
import uvicorn

from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Role,
)
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.apps.starlette_app import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils.message import new_agent_text_message

# OpenAI Agents SDK
from agents import Agent, RunConfig, Runner, function_tool
from agents.models.multi_provider import MultiProvider

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from dotenv import load_dotenv


# Optional observability imports
try:
    import phoenix as px
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("⚠️  Phoenix not available - observability features will be limited")
    print("💡 Install with: uv add arize-phoenix openinference-instrumentation-openai")

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    print("⚠️  Logfire not available - some tracing features disabled")
    print("💡 Install with: uv add logfire")


load_dotenv()

# Apply nest_asyncio for compatibility
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "gemma-3-4b-it"


# Configure observability
def setup_observability():
    """Setup OpenTelemetry and Phoenix observability"""

    # Setup basic OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    phoenix_session = None

    # Skip console exporter - we'll use Phoenix only

    # Setup Phoenix if available
    phoenix_connected = False
    if PHOENIX_AVAILABLE:
        phoenix_endpoint = os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"
        )

        try:
            print(f"🔍 Testing Phoenix connectivity at {phoenix_endpoint}...")

            # Quick connectivity test for Phoenix
            import httpx

            # Extract base URL for health check
            if ":6006" in phoenix_endpoint:
                health_url = phoenix_endpoint.replace("/v1/traces", "")
            else:
                health_url = phoenix_endpoint.replace("/v1/traces", "")

            # Test connectivity with short timeout
            with httpx.Client(timeout=1.0) as test_client:
                try:
                    response = test_client.get(f"{health_url}/health")
                    if response.status_code == 200:
                        phoenix_connected = True
                        print("✅ Phoenix health check passed")
                except httpx.RequestError:
                    try:
                        response = test_client.get(health_url)
                        if response.status_code in [200, 404]:
                            phoenix_connected = True
                            print("✅ Phoenix server responding")
                    except httpx.RequestError:
                        phoenix_connected = False
                        print("❌ Phoenix server not reachable")

            if phoenix_connected:
                # Setup Phoenix OTLP exporter
                phoenix_exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
                span_processor = BatchSpanProcessor(phoenix_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)

                # Instrument OpenAI with Phoenix
                OpenAIInstrumentor().instrument()
                print(f"📊 Phoenix observability enabled: {phoenix_endpoint}")
                print("✅ OpenAI instrumentation enabled")
            else:
                print("📊 Phoenix not reachable - using console tracing only")
                print("💡 To enable Phoenix observability:")
                print(
                    "   🐳 Docker: docker run -p 6006:6006 -p 4318:4318 arizephoenix/phoenix:latest"
                )
                print("   🚀 Compose: docker-compose up phoenix")

        except ImportError:
            print("⚠️  httpx not available - cannot test Phoenix connectivity")
            phoenix_connected = False
        except Exception as e:
            print(f"⚠️  Phoenix setup failed: {e}")
            print("📊 Falling back to console-only tracing")
            phoenix_connected = False
    else:
        print("📊 Phoenix not available - using basic tracing")
        print(
            "💡 To enable Phoenix: uv add arize-phoenix openinference-instrumentation-openai"
        )

    # Setup OTLP exporter (optional - for external observability platforms)
    external_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if external_endpoint:
        try:
            print(f"🔍 Setting up external OTLP exporter: {external_endpoint}")
            otlp_exporter = OTLPSpanExporter(
                endpoint=external_endpoint,
                headers={
                    "Authorization": f"Bearer {os.getenv('OTEL_EXPORTER_OTLP_HEADERS', '')}"
                },
            )
            external_span_processor = SimpleSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(external_span_processor)
            print("✅ External OTLP exporter configured")
        except Exception as e:
            print(f"⚠️  External OTLP exporter setup failed: {e}")

    # Setup Logfire for OpenAI Agents if available
    if LOGFIRE_AVAILABLE and os.getenv("LOGFIRE_TOKEN"):
        try:
            logfire.configure(
                token=os.getenv("LOGFIRE_TOKEN"), project_name="a2a-openai-demo"
            )
            print("✅ Logfire configured successfully")
        except Exception as e:
            print(f"⚠️  Logfire setup failed: {e}")

    # Summary of observability status
    if PHOENIX_AVAILABLE and phoenix_connected:
        print("📊 Phoenix observability: ✅ Connected")
        print(f"📊 Phoenix UI: http://127.0.0.1:6006")
    else:
        print("📊 Phoenix observability: ❌ Not Available")

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
                "✅ Validation successful: Resource validation successful",
                "✅ Validation successful: Connection test passed",
                "✅ Validation successful: Endpoint is accessible",
                "✅ Validation successful: Service is responding correctly",
                "✅ Validation successful: Test validation completed successfully",
            ]

            result = random.choice(sample_results)
            span.set_attributes({"validation_success": True, "random_result": True})
            return result
        else:
            result = f"❌ Validation failed: Unable to validate {test_url}"
            span.set_attributes({"validation_success": False, "random_result": False})
            return result


class URLValidationAgentWrapper:
    """Agent specifically for URL validation using random_testing_tool"""

    def __init__(self):
        # Initialize agent with only the random testing tool
        self.agent = Agent(
            name="URL Validation Agent",
            model=MODEL_NAME,
            tools=[random_testing_tool],
            instructions="""You are a URL validation assistant. Your job is to:

1. When a user provides input, check if it contains a valid URL format (starts with http:// or https://)
2. If it's a valid URL format, use the random_testing_tool to validate it
3. If the validation is successful, respond with: "VALIDATION_SUCCESS: Your URL has been validated and you can now proceed."
4. If the validation fails or the input is not a URL, respond with: "VALIDATION_FAILED: Please provide a valid URL starting with http:// or https://"

Always start your response with either "VALIDATION_SUCCESS:" or "VALIDATION_FAILED:" for easy parsing.""",
        )

        # Initialize agent run config
        self.runConfig = RunConfig(
            model_provider=MultiProvider(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
                openai_use_responses=os.getenv("USE_RESPONSES_API", "true").lower()
                == "true",
            )
        )

    async def invoke(self, message: str) -> tuple[str, bool]:
        """Invoke the validation agent and return (response, is_validated)"""

        with tracer.start_as_current_span("url_validation_agent_execution") as span:
            span.set_attributes(
                {
                    "agent_model": MODEL_NAME,
                    "user_message": message[:200],
                    "span.kind": "llm",
                    "agent_type": "url_validation",
                }
            )

            try:
                # Run the validation agent
                result = await Runner.run(
                    self.agent, message, run_config=self.runConfig
                )
                agent_response = str(result)

                # Check if validation was successful by looking for success indicators
                is_validated = "VALIDATION_SUCCESS" in agent_response

                span.set_attributes(
                    {
                        "agent_response": agent_response[:200],
                        "llm.response.model": MODEL_NAME,
                        "url_validated": is_validated,
                    }
                )

                return agent_response, is_validated

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", str(e))
                agent_response = f"I encountered an error during validation: {str(e)}"
                return agent_response, False


class OpenAIAgentWrapper:
    """Wrapper for OpenAI agent to work with A2A framework"""

    def __init__(self):
        # Initialize OpenAI Agent with tools (excluding random_testing_tool)
        self.agent = Agent(
            name="Assistant Agent",
            model="gemma-3-4b-it",
            tools=[get_weather, calculate],
            instructions="""You are a helpful AI assistant with access to weather and calculation tools.
            
            You can:
            1. Get weather information for any location using the get_weather tool
            2. Perform basic math calculations using the calculate tool
            
            Always be helpful and provide clear, accurate responses.""",
        )

        # Initialize agent run config
        self.runConfig = RunConfig(
            model_provider=MultiProvider(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
                openai_use_responses=os.getenv("USE_RESPONSES_API", "true").lower()
                == "true",
            )
        )

    async def invoke(self, message: str) -> str:
        """Invoke the OpenAI agent with a message"""

        with tracer.start_as_current_span("openai_agent_execution") as span:
            span.set_attributes(
                {
                    "agent_model": MODEL_NAME,
                    "user_message": message[:200],
                    "span.kind": "llm",
                    "agent_type": "main_agent",
                }
            )

            try:
                # Run the agent - OpenAI instrumentation will automatically trace this
                result = await Runner.run(
                    self.agent, message, run_config=self.runConfig
                )
                agent_response = str(result)

                span.set_attribute("agent_response", agent_response[:200])
                span.set_attribute("llm.response.model", MODEL_NAME)

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
        self.main_agent = OpenAIAgentWrapper()
        self.validation_agent = URLValidationAgentWrapper()
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def cancel(self, context: RequestContext) -> None:
        """Cancel the current agent execution

        Args:
            context: The request context to cancel
        """
        with tracer.start_as_current_span("a2a_agent_cancel") as span:
            session_id = self.get_session_id(context)

            span.set_attributes(
                {
                    "session_id": session_id,
                    "operation": "cancel",
                    "session_exists": session_id in self.sessions,
                }
            )

            # Clean up session if it exists
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session["cancelled_at"] = datetime.now(timezone.utc).isoformat()
                session["state"] = "cancelled"

                span.set_attributes(
                    {
                        "previous_state": session.get("state", "unknown"),
                        "cleanup_performed": True,
                    }
                )

                print(f"🛑 Cancelled session {session_id}")
            else:
                span.set_attribute("cleanup_performed", False)
                print(f"🛑 Attempted to cancel non-existent session {session_id}")

    def get_session_id(self, context: RequestContext) -> str:
        """Extract or generate session ID from context"""
        # Try to get session ID from context attributes
        session_id = getattr(context, "session_id", None)
        if not session_id:
            # Generate a new session ID based on context info
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        return session_id

    def initialize_session(self, session_id: str, context: RequestContext) -> None:
        """Initialize a new session with default state"""
        self.sessions[session_id] = {
            "state": "validation",  # validation, running
            "context_id": getattr(context, "context_id", None),
            "task_id": getattr(context, "task_id", None),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validated_url": None,
            "message_history": [],
        }

    def update_session_state(self, session_id: str, new_state: str, **kwargs) -> None:
        """Update session state and additional data"""
        if session_id in self.sessions:
            self.sessions[session_id]["state"] = new_state
            self.sessions[session_id]["updated_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            for key, value in kwargs.items():
                self.sessions[session_id][key] = value

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the agent with full observability and session management"""

        with tracer.start_as_current_span("a2a_agent_execute") as span:
            try:
                # Get session ID and initialize if needed
                session_id = self.get_session_id(context)

                if session_id not in self.sessions:
                    self.initialize_session(session_id, context)

                session = self.sessions[session_id]

                # Extract message from RequestContext using get_user_input()
                text_content = context.get_user_input()
                print(f"🔍 DEBUG: User input: {text_content}")

                span.set_attributes(
                    {
                        "message_text": text_content[:200],
                        "session_id": session_id,
                        "session_state": session["state"],
                        "input.value": text_content,
                        "session.id": session_id,
                        "context_type": str(type(context)),
                        "extraction_method": "get_user_input",
                        "extraction_success": True,
                    }
                )

                # Add message to history
                session["message_history"].append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": Role.user,
                        "content": text_content,
                    }
                )

                response = ""

                # Handle different session states using agents
                if session["state"] == "validation":
                    # Use validation agent to handle URL validation
                    response, is_validated = await self.validation_agent.invoke(
                        text_content
                    )

                    if is_validated:
                        # URL validated successfully, switch to main agent
                        self.update_session_state(
                            session_id, "running", validated_url=text_content
                        )

                        # Add a note that they can now ask questions
                        response += "\n\n Great! Now I can help you with weather information, calculations, and other questions. What would you like to know?"

                # if running, use main agent
                if session["state"] == "running":
                    # Use main agent for regular interactions
                    url_context = f"Note: User previously validated URL: {session.get('validated_url', 'N/A')}\n\n"
                    full_message = url_context + text_content

                    response = await self.main_agent.invoke(full_message)

                # Add response to history
                session["message_history"].append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": Role.agent,
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
                print(f"🔍 DEBUG: Exception in execute: {e}")
                print(f"🔍 DEBUG: Exception type: {type(e)}")

                error_response = f"I encountered an error: {str(e)}"
                event_queue.enqueue_event(new_agent_text_message(error_response))


def create_agent_card() -> AgentCard:
    """Create the agent card with capabilities and skills"""

    return AgentCard(
        id="openai-validation-weather-math-agent",
        name="OpenAI URL Validation & Assistant Agent",
        description="A multi-agent AI system with URL validation and assistant capabilities",
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
                id="url_validation",
                name="URL Validation",
                description="Validate URLs using intelligent testing and provide feedback",
                examples=[
                    "https://example.com/test",
                    "Please validate this URL: https://google.com",
                ],
                tags=["validation", "url", "testing"],
            ),
            AgentSkill(
                id="get_weather",
                name="Get Weather",
                description="Get current weather for any location (available after URL validation)",
                examples=["What's the weather in Tokyo?", "Weather in New York"],
                tags=["weather", "location", "forecast"],
            ),
            AgentSkill(
                id="calculate",
                name="Calculate",
                description="Perform basic mathematical operations (available after URL validation)",
                examples=["Calculate 15 * 7", "What's 100 / 5?"],
                tags=["math", "calculation", "arithmetic"],
            ),
        ],
    )


async def main():
    """Main function to run the A2A server"""

    print("🚀 Starting A2A Server with Multi-Agent System and Phoenix Observability")
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
    print(
        f"  • Phoenix: {'✅ Enabled' if PHOENIX_AVAILABLE else '⚠️  Not Available'} (UI at http://127.0.0.1:6006)"
    )
    print(
        f"  • OpenAI Instrumentation: {'✅ Enabled' if PHOENIX_AVAILABLE else '⚠️  Limited'}"
    )
    print(
        f"  • Logfire: {'✅ Enabled' if LOGFIRE_AVAILABLE and os.getenv('LOGFIRE_TOKEN') else '⚠️  Disabled'}"
    )

    if not PHOENIX_AVAILABLE:
        print("\n💡 To enable full observability features:")
        print("   pip install phoenix openinference-instrumentation-openai")

    if not LOGFIRE_AVAILABLE:
        print("\n💡 To enable Logfire:")
        print("   pip install logfire")

    # Create agent card
    agent_card = create_agent_card()

    # Create agent executor
    agent_executor = ObservableAgentExecutor()

    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("\n🌐 Server starting on http://localhost:8000")
    print("📡 A2A RPC endpoint: http://localhost:8000/")
    print("📋 Agent card: http://localhost:8000/.well-known/agent.json")
    print("📊 Phoenix UI: http://127.0.0.1:6006")
    print("\n🤖 Multi-Agent System:")
    print("  • URL Validation Agent - Handles URL validation with random_testing_tool")
    print("  • Main Assistant Agent - Weather & calculation tools (post-validation)")
    print("\n📋 Session Flow:")
    print("  1️⃣  Validation Phase: Agent validates URLs intelligently")
    print("  2️⃣  Running Phase: Full assistant capabilities unlocked")
    print("\n💡 Example interactions:")
    print("  • Start: 'https://example.com/test' (validation agent)")
    print("  • After validation: 'What's the weather in Tokyo?' (main agent)")
    print("\n📈 All agent interactions, tool calls, and state transitions")
    print(
        "   will be automatically traced and visible in Phoenix at http://127.0.0.1:6006"
    )
    print("\nPress Ctrl+C to stop the server")

    try:
        # Run the server
        config = uvicorn.Config(
            app=app.build(), host="0.0.0.0", port=8000, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
