"""
A2A Client with Full Observability (Using Official Google A2A SDK)
This implements a complete A2A client using the official a2a-sdk that calls the A2A server
with tracing for all HTTP calls.
"""

import asyncio
import os
import uuid
from typing import Dict, Any

import nest_asyncio

# Official Google A2A SDK imports
from a2a.client import A2AClient
from a2a.types import Message, TextPart, Role

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Langfuse imports
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# Apply nest_asyncio for compatibility
nest_asyncio.apply()


# Configure observability
def setup_observability():
    """Setup OpenTelemetry and Langfuse observability for the client"""

    # Setup OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Setup OTLP exporter (optional)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers={
                "Authorization": f"Bearer {os.getenv('OTEL_EXPORTER_OTLP_HEADERS', '')}"
            },
        )
        span_processor = SimpleSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

    # Setup Langfuse
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    # Instrument HTTPX for HTTP call tracing
    HTTPXClientInstrumentor().instrument()

    return tracer, langfuse


# Initialize observability
tracer, langfuse = setup_observability()


class ObservableA2AClient:
    """A2A Client with full observability using the official Google A2A SDK"""

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.client = A2AClient(base_url=server_url)
        self.session_id = str(uuid.uuid4())

        print("🔗 A2A Client initialized")
        print(f"📡 Server URL: {server_url}")
        print(f"🆔 Session ID: {self.session_id}")

    @observe(name="a2a_send_message")
    async def send_message(self, text: str) -> str:
        """Send a message to the A2A server with full observability"""

        with tracer.start_as_current_span("a2a_client_send_message") as span:
            span.set_attributes(
                {
                    "server_url": self.server_url,
                    "session_id": self.session_id,
                    "message_text": text[:200],  # Truncate for logging
                    "message_length": len(text),
                }
            )

            # Log to Langfuse
            langfuse_context.update_current_trace(
                name="A2A Client Message",
                input=text,
                metadata={"server_url": self.server_url, "session_id": self.session_id},
            )

            try:
                # Create A2A message
                message = Message(
                    role=Role.USER,
                    parts=[TextPart(type="text", text=text)],
                )

                span.set_attribute("message_created", True)

                # Send message via A2A SDK
                with tracer.start_as_current_span("a2a_sdk_call") as sdk_span:
                    sdk_span.set_attributes(
                        {
                            "operation": "send_message",
                            "message_parts": len(message.parts),
                        }
                    )

                    response = await self.client.send_message(message=message)

                    sdk_span.set_attribute("response_received", True)

                # Extract response text
                response_text = ""
                if hasattr(response, "message") and response.message:
                    for part in response.message.parts:
                        if hasattr(part, "text"):
                            response_text += part.text

                span.set_attributes(
                    {
                        "response_received": True,
                        "response_text": response_text[:200],  # Truncate for logging
                        "response_length": len(response_text),
                    }
                )

                # Update Langfuse with response
                langfuse_context.update_current_trace(output=response_text)

                return response_text

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", str(e))

                # Log error to Langfuse
                langfuse_context.update_current_trace(
                    output=f"Error: {str(e)}", metadata={"error": str(e)}
                )

                raise e

    @observe(name="a2a_get_agent_card")
    async def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card from the A2A server"""

        with tracer.start_as_current_span("a2a_client_get_agent_card") as span:
            span.set_attribute("server_url", self.server_url)

            try:
                # Get agent card via A2A SDK
                card = await self.client.get_agent_card()

                span.set_attributes(
                    {
                        "agent_card_received": True,
                        "agent_name": getattr(card, "name", "unknown"),
                        "agent_version": getattr(card, "version", "unknown"),
                    }
                )

                # Convert to dict for JSON serialization
                if hasattr(card, "model_dump"):
                    return card.model_dump()
                else:
                    return {
                        "name": getattr(card, "name", "Unknown Agent"),
                        "description": getattr(card, "description", "No description"),
                        "version": getattr(card, "version", "Unknown"),
                        "skills": getattr(card, "skills", []),
                    }

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", str(e))
                raise e


async def interactive_demo():
    """Interactive demo of the A2A client"""

    print("🎯 A2A Client Demo with Full Observability")
    print("=" * 50)

    # Check if server is likely running
    server_url = "http://localhost:8000"
    print(f"🔍 Connecting to A2A server at {server_url}")

    try:
        # Create client
        client = ObservableA2AClient(server_url)

        # Get agent card
        print("\n📋 Getting agent information...")
        agent_card = await client.get_agent_card()
        print(f"✅ Connected to: {agent_card.get('name', 'Unknown Agent')}")
        print(f"📝 Description: {agent_card.get('description', 'No description')}")
        print(f"🔧 Skills available: {len(agent_card.get('skills', []))}")

        # Interactive chat loop
        print("\n💬 Interactive Chat (type 'quit' to exit)")
        print("💡 Try: 'What's the weather in Tokyo?' or 'Calculate 15 * 7'")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("🤖 Agent: ", end="", flush=True)

                # Send message with observability
                response = await client.send_message(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to connect to A2A server: {e}")
        print("💡 Make sure the A2A server is running on http://localhost:8000")


async def automated_demo():
    """Automated demo showing various A2A interactions"""

    print("🤖 Automated A2A Demo with Full Observability")
    print("=" * 50)

    try:
        client = ObservableA2AClient("http://localhost:8000")

        # Test messages
        test_messages = [
            "Hello! Can you introduce yourself?",
            "What's the weather like in New York?",
            "Can you calculate 25 * 4?",
            "What's the weather in Tokyo and London?",
            "Calculate the result of 100 divided by 5, then multiply by 3",
        ]

        print(f"\n🧪 Running {len(test_messages)} test interactions...")

        for i, message in enumerate(test_messages, 1):
            print(f"\n📤 Test {i}: {message}")

            try:
                response = await client.send_message(message)
                print(f"📥 Response: {response}")

                # Small delay between requests
                await asyncio.sleep(1)

            except Exception as e:
                print(f"❌ Error in test {i}: {e}")

        print("\n✅ Automated demo completed!")
        print("📊 Check your observability dashboard for traces and metrics")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def main():
    """Main function"""

    print("🚀 A2A Client with Official Google A2A SDK and Full Observability")
    print("=" * 60)

    # Check observability setup
    observability_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing_obs = [var for var in observability_vars if not os.getenv(var)]

    if missing_obs:
        print(f"⚠️  Optional observability variables not set: {missing_obs}")
        print("Observability features will be limited.")

    print("\n📊 Observability Features:")
    print("  • OpenTelemetry: ✅ Enabled")
    print(f"  • Langfuse: {'✅ Enabled' if not missing_obs else '⚠️  Limited'}")
    print("  • HTTP Tracing: ✅ Enabled")

    # Choose demo mode
    print("\n🎯 Choose demo mode:")
    print("1. Interactive chat")
    print("2. Automated demo")

    try:
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == "1":
            await interactive_demo()
        elif choice == "2":
            await automated_demo()
        else:
            print("❌ Invalid choice. Running interactive demo...")
            await interactive_demo()

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
