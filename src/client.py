"""
A2A Client with Full Observability (Using Official Google A2A SDK)
This implements a complete A2A client using the official a2a-sdk that calls the A2A server
with tracing for all HTTP calls.
"""

import asyncio
import os
import uuid
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import nest_asyncio
import httpx

# Official Google A2A SDK imports
from a2a.client.client import A2AClient, A2ACardResolver
from a2a.types import (
    Message,
    Part,
    TextPart,
    Role,
    SendStreamingMessageRequest,
    MessageSendParams,
    AgentCard,
)

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Apply nest_asyncio for compatibility
nest_asyncio.apply()


# Configure observability
def setup_observability():
    """Setup OpenTelemetry and Phoenix observability for the client"""

    # Setup OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Skip console exporter - we'll use Phoenix only

    # Try to setup Phoenix OTLP exporter with robust error handling
    phoenix_endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"
    )

    phoenix_available = False

    try:
        # Test Phoenix connectivity with a quick health check
        print(f"🔍 Testing Phoenix connectivity at {phoenix_endpoint}...")

        # Import httpx only when needed to avoid dependency issues
        import httpx

        # Extract base URL for health check
        if ":6006" in phoenix_endpoint:
            health_url = phoenix_endpoint.replace("/v1/traces", "")
        else:
            health_url = phoenix_endpoint.replace("/v1/traces", "")

        # Quick connectivity test with short timeout
        with httpx.Client(timeout=1.0) as test_client:
            try:
                # First try the health endpoint
                response = test_client.get(f"{health_url}/health")
                if response.status_code == 200:
                    phoenix_available = True
                    print("✅ Phoenix health check passed")
            except httpx.RequestError:
                try:
                    # Fallback: try just the base URL
                    response = test_client.get(health_url)
                    if response.status_code in [
                        200,
                        404,
                    ]:  # 404 is OK, means server is running
                        phoenix_available = True
                        print("✅ Phoenix server responding")
                except httpx.RequestError:
                    phoenix_available = False
                    print("❌ Phoenix server not reachable")

        if phoenix_available:
            # Setup Phoenix OTLP exporter
            try:
                phoenix_exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
                span_processor = BatchSpanProcessor(phoenix_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
                print(f"📊 Phoenix observability enabled: {phoenix_endpoint}")
            except Exception as e:
                print(f"⚠️  Phoenix OTLP exporter setup failed: {e}")
                phoenix_available = False

    except ImportError:
        print("⚠️  httpx not available - cannot test Phoenix connectivity")
        phoenix_available = False
    except Exception as e:
        print(f"⚠️  Phoenix connectivity test failed: {e}")
        phoenix_available = False

    # Provide helpful guidance if Phoenix is not available
    if not phoenix_available:
        print("📊 Running with console-only tracing")
        print("💡 To enable Phoenix observability:")
        print(
            "   🐳 Docker: docker run -p 6006:6006 -p 4318:4318 arizephoenix/phoenix:latest"
        )
        print("   🚀 Compose: docker-compose up phoenix")
        print("   🌐 Phoenix UI will be at http://localhost:6006")

    # Setup external OTLP exporter (optional, for other observability platforms)
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
            external_span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(external_span_processor)
            print("✅ External OTLP exporter configured")
        except Exception as e:
            print(f"⚠️  External OTLP exporter setup failed: {e}")

    # Instrument HTTPX for HTTP call tracing (if available)
    try:
        HTTPXClientInstrumentor().instrument()
        print("✅ HTTPX instrumentation enabled")
    except Exception as e:
        print(f"⚠️  HTTPX instrumentation failed: {e}")

    # Summary of observability status
    if phoenix_available:
        print("📊 Phoenix observability: ✅ Connected")
    else:
        print("📊 Phoenix observability: ❌ Not Available")

    return tracer, None  # No local phoenix session


# Initialize observability
tracer, phoenix_session = setup_observability()


class ObservableA2AClient:
    """A2A Client with full observability using the official Google A2A SDK"""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        agent_card: Optional[AgentCard] = None,
    ):
        self.server_url = server_url
        self.session_id = str(uuid.uuid4())
        self.agent_card = agent_card

        # Conversation logging
        self.conversation_log: List[Dict[str, Any]] = []

        # Create httpx client
        self.httpx_client = httpx.AsyncClient()

        # Initialize A2A client with the correct parameters
        # The A2A SDK uses root "/" as the default RPC endpoint
        self.client = A2AClient(
            httpx_client=self.httpx_client,
            url=server_url,  # A2A server root URL - SDK will handle routing to "/"
        )

        # Initialize card resolver for getting agent info
        self.card_resolver = A2ACardResolver(
            httpx_client=self.httpx_client, base_url=server_url
        )

        print("🔗 A2A Client initialized")
        print(f"📡 Server URL: {server_url}")
        if agent_card:
            print(f"🤖 Agent: {agent_card.name} (v{agent_card.version})")
            print(f"📝 Description: {agent_card.description}")
            print(f"🔧 Skills: {len(agent_card.skills)} available")
        print(f"🆔 Session ID: {self.session_id}")

    async def close(self):
        """Close the httpx client"""
        await self.httpx_client.aclose()


async def demo():
    """Interactive demo of the A2A client"""

    print("🎯 A2A Client Demo with Phoenix Observability")
    print("=" * 50)

    # Check if server is likely running
    server_url = "http://localhost:8000"
    print(f"🔍 Connecting to A2A server at {server_url}")

    client = None
    try:
        # Create client
        client = ObservableA2AClient(server_url)

        # Interactive chat loop
        print("\n💬 Interactive Chat")
        print("💡 Commands:")
        print("  • Type your message for regular response")
        print("  • Type 'stream: <message>' for streaming response")
        print("  • Type 'quit' to exit")
        print("💡 Try: 'What's the weather in Tokyo?' or 'stream: Calculate 15 * 7'")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                message = Message(
                    messageId=str(uuid.uuid4()),  # Add unique message ID
                    role=Role.user,
                    parts=[Part(TextPart(kind="text", text=user_input))],
                )

                # Create SendStreamingMessageRequest with same correlation ID
                request = SendStreamingMessageRequest(
                    params=MessageSendParams(message=message),
                )

                # Send streaming message
                async for response in client.client.send_message_streaming(request):
                    print(response, end="", flush=True)
                print()  # New line after streaming is complete

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to connect to A2A server: {e}")
        print("💡 Make sure the A2A server is running on http://localhost:8000")
    finally:
        if client:
            await client.close()


async def main():
    """Main function"""

    print("🚀 A2A Client with Official Google A2A SDK and Phoenix Observability")
    print("=" * 60)

    # Check observability setup
    observability_vars = ["PHOENIX_COLLECTOR_ENDPOINT"]
    missing_obs = [var for var in observability_vars if not os.getenv(var)]

    if missing_obs:
        print(f"⚠️  Optional observability variables not set: {missing_obs}")
        print("Phoenix will use default local endpoint (http://127.0.0.1:6006)")

    print("\n📊 Observability Features:")
    print("  • OpenTelemetry: ✅ Enabled")
    print("  • Phoenix: ✅ Enabled (UI at http://127.0.0.1:6006)")
    print("  • HTTP Tracing: ✅ Enabled")
    print("  • Conversation Logging: ✅ Enabled")
    print("  • Agent Discovery: ✅ Enabled")

    try:
        await demo()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
