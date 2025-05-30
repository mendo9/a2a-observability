"""
A2A Client with Full Observability (Using Official Google A2A SDK)
This implements a complete A2A client using the official a2a-sdk that calls the A2A server
with tracing for all HTTP calls.
"""

import asyncio
import os
import uuid
from typing import Dict, Any, List, Optional
import socket
import json
from urllib.parse import urlparse

import nest_asyncio
import httpx

# Official Google A2A SDK imports
from a2a.client.client import A2AClient, A2ACardResolver
from a2a.types import (
    Message,
    TextPart,
    Role,
    SendMessageRequest,
    SendStreamingMessageRequest,
    MessageSendParams,
    AgentCard,
)

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Phoenix imports
import phoenix as px
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# Apply nest_asyncio for compatibility
nest_asyncio.apply()


# Configure observability
def setup_observability():
    """Setup OpenTelemetry and Phoenix observability for the client"""

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

    # Setup OTLP exporter (optional)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers={
                "Authorization": f"Bearer {os.getenv('OTEL_EXPORTER_OTLP_HEADERS', '')}"
            },
        )
        external_span_processor = SimpleSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(external_span_processor)

    # Instrument HTTPX for HTTP call tracing
    HTTPXClientInstrumentor().instrument()

    return tracer, phoenix_session


# Initialize observability
tracer, phoenix_session = setup_observability()


class AgentDiscovery:
    """Agent discovery service for finding A2A agents automatically"""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.discovered_agents: List[Dict[str, Any]] = []

    async def discover_by_well_known(
        self, hosts: List[str], ports: List[int] = None
    ) -> List[Dict[str, Any]]:
        """Discover agents by checking well-known endpoints on specified hosts"""
        if ports is None:
            ports = [8000, 8080, 3000, 5000, 9000]  # Common ports for dev servers

        discovered = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for host in hosts:
                for port in ports:
                    try:
                        # Try both http and https
                        for protocol in ["http", "https"]:
                            url = f"{protocol}://{host}:{port}/.well-known/agent.json"

                            try:
                                response = await client.get(url)
                                if response.status_code == 200:
                                    agent_data = response.json()

                                    # Validate it's a proper agent card
                                    try:
                                        agent_card = AgentCard.model_validate(
                                            agent_data
                                        )
                                        discovered.append(
                                            {
                                                "agent_card": agent_card,
                                                "base_url": f"{protocol}://{host}:{port}",
                                                "discovery_method": "well-known",
                                                "response_time": response.elapsed.total_seconds(),
                                            }
                                        )
                                        print(
                                            f"🔍 Discovered agent: {agent_card.name} at {protocol}://{host}:{port}"
                                        )
                                        break  # Found agent, don't try https if http worked

                                    except Exception as e:
                                        print(f"⚠️  Invalid agent card at {url}: {e}")

                            except httpx.RequestError:
                                continue  # Try next protocol/port

                    except Exception as e:
                        continue  # Try next host/port

        return discovered

    async def discover_local_network(self) -> List[Dict[str, Any]]:
        """Discover agents on the local network"""
        # Get local network range
        local_hosts = self._get_local_network_hosts()
        print(f"🌐 Scanning {len(local_hosts)} hosts on local network...")

        return await self.discover_by_well_known(
            local_hosts[:20]
        )  # Limit to first 20 for speed

    async def discover_localhost(self) -> List[Dict[str, Any]]:
        """Discover agents running on localhost"""
        print("🏠 Scanning localhost for A2A agents...")
        return await self.discover_by_well_known(["localhost", "127.0.0.1"])

    def _get_local_network_hosts(self) -> List[str]:
        """Get list of potential hosts on the local network"""
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            # Generate network range (simple /24 subnet)
            ip_parts = local_ip.split(".")
            network_base = ".".join(ip_parts[:3])

            hosts = [f"{network_base}.{i}" for i in range(1, 255)]
            return hosts

        except Exception:
            return ["127.0.0.1"]  # Fallback to localhost only

    async def discover_by_registry(
        self, registry_urls: List[str]
    ) -> List[Dict[str, Any]]:
        """Discover agents through agent registries (future enhancement)"""
        # Placeholder for registry-based discovery
        # Could integrate with agent marketplaces, service discovery systems, etc.
        discovered = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for registry_url in registry_urls:
                try:
                    response = await client.get(f"{registry_url}/api/agents")
                    if response.status_code == 200:
                        agents_data = response.json()
                        for agent_data in agents_data.get("agents", []):
                            try:
                                agent_card = AgentCard.model_validate(
                                    agent_data["agent_card"]
                                )
                                discovered.append(
                                    {
                                        "agent_card": agent_card,
                                        "base_url": agent_data["base_url"],
                                        "discovery_method": "registry",
                                        "registry": registry_url,
                                    }
                                )
                            except Exception as e:
                                print(
                                    f"⚠️  Invalid agent in registry {registry_url}: {e}"
                                )

                except Exception as e:
                    print(f"❌ Failed to query registry {registry_url}: {e}")

        return discovered


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

    @classmethod
    async def discover(
        cls,
        discovery_scope: str = "localhost",
        agent_filter: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> List["ObservableA2AClient"]:
        """Discover and connect to A2A agents automatically

        Args:
            discovery_scope: "localhost", "local-network", "registry", or "all"
            agent_filter: Filter criteria for agents (e.g., {"skills": ["weather"]})
            timeout: Discovery timeout in seconds

        Returns:
            List of connected A2A clients
        """
        discovery = AgentDiscovery(timeout=timeout)
        discovered_agents = []

        print(f"🔍 Starting agent discovery (scope: {discovery_scope})...")

        if discovery_scope in ["localhost", "all"]:
            discovered_agents.extend(await discovery.discover_localhost())

        if discovery_scope in ["local-network", "all"]:
            discovered_agents.extend(await discovery.discover_local_network())

        if discovery_scope in ["registry", "all"]:
            # Example registry URLs (these would be real registry services)
            registry_urls = [
                "https://a2a-registry.example.com",
                "https://agent-marketplace.example.com",
            ]
            discovered_agents.extend(
                await discovery.discover_by_registry(registry_urls)
            )

        # Filter agents if criteria provided
        if agent_filter:
            discovered_agents = cls._filter_agents(discovered_agents, agent_filter)

        # Create clients for discovered agents
        clients = []
        for agent_info in discovered_agents:
            try:
                client = cls(
                    server_url=agent_info["base_url"],
                    agent_card=agent_info["agent_card"],
                )
                clients.append(client)

            except Exception as e:
                print(f"❌ Failed to create client for {agent_info['base_url']}: {e}")

        print(f"✅ Discovery complete: {len(clients)} agents available")
        return clients

    @classmethod
    async def discover_and_select(
        cls,
        discovery_scope: str = "localhost",
        agent_filter: Optional[Dict[str, Any]] = None,
        auto_select: bool = False,
    ) -> Optional["ObservableA2AClient"]:
        """Discover agents and let user select one"""

        clients = await cls.discover(discovery_scope, agent_filter)

        if not clients:
            print("❌ No agents discovered")
            return None

        if len(clients) == 1 or auto_select:
            selected_client = clients[0]
            print(f"🎯 Auto-selected: {selected_client.agent_card.name}")
            return selected_client

        # Let user choose
        print(f"\n🤖 Discovered {len(clients)} agents:")
        for i, client in enumerate(clients):
            card = client.agent_card
            print(f"  {i + 1}. {card.name} (v{card.version})")
            print(f"     📝 {card.description}")
            print(f"     🌐 {client.server_url}")
            print(f"     🔧 {len(card.skills)} skills available")
            print()

        try:
            choice = int(input(f"Select agent (1-{len(clients)}): ")) - 1
            if 0 <= choice < len(clients):
                selected_client = clients[choice]
                print(f"🎯 Selected: {selected_client.agent_card.name}")

                # Close unused clients
                for i, client in enumerate(clients):
                    if i != choice:
                        await client.close()

                return selected_client
            else:
                print("❌ Invalid selection")
                return None

        except (ValueError, KeyboardInterrupt):
            print("❌ Selection cancelled")
            # Close all clients
            for client in clients:
                await client.close()
            return None

    @staticmethod
    def _filter_agents(
        agents: List[Dict[str, Any]], filter_criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter discovered agents based on criteria"""
        filtered = []

        for agent_info in agents:
            agent_card = agent_info["agent_card"]
            matches = True

            # Filter by skills
            if "skills" in filter_criteria:
                required_skills = filter_criteria["skills"]
                agent_skill_ids = [skill.id for skill in agent_card.skills]
                if not any(skill in agent_skill_ids for skill in required_skills):
                    matches = False

            # Filter by capabilities
            if "capabilities" in filter_criteria:
                required_caps = filter_criteria["capabilities"]
                if "streaming" in required_caps:
                    if not agent_card.capabilities.streaming:
                        matches = False

            # Filter by name pattern
            if "name_pattern" in filter_criteria:
                pattern = filter_criteria["name_pattern"].lower()
                if pattern not in agent_card.name.lower():
                    matches = False

            if matches:
                filtered.append(agent_info)

        return filtered

    def _log_conversation_item(
        self, item_type: str, content: str, metadata: Dict[str, Any] = None
    ):
        """Log a conversation item with timestamp and metadata"""
        from datetime import datetime

        log_item = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "type": item_type,  # "user_message", "agent_response", "status_update", "artifact"
            "content": content,
            "metadata": metadata or {},
        }

        self.conversation_log.append(log_item)

        # Print with color coding
        timestamp = log_item["timestamp"][:19]  # Remove microseconds
        if item_type == "user_message":
            print(f"📝 [{timestamp}] USER: {content}")
        elif item_type == "agent_response":
            print(f"🤖 [{timestamp}] AGENT: {content}")
        elif item_type == "status_update":
            print(f"📊 [{timestamp}] STATUS: {content}")
        elif item_type == "artifact":
            print(f"📎 [{timestamp}] ARTIFACT: {content}")
        elif item_type == "stream_chunk":
            print(f"⚡ [{timestamp}] STREAM: {content}")

    async def send_message(self, text: str) -> str:
        """Send a message to the A2A server with full observability"""

        # Log user message
        self._log_conversation_item("user_message", text)

        with tracer.start_as_current_span("a2a_client_send_message") as span:
            # Generate a single correlation ID for this message exchange
            correlation_id = str(uuid.uuid4())

            span.set_attributes(
                {
                    "server_url": self.server_url,
                    "session_id": self.session_id,
                    "correlation_id": correlation_id,
                    "message_text": text[:200],  # Truncate for logging
                    "message_length": len(text),
                    "input.value": text,
                    "session.id": self.session_id,
                }
            )

            try:
                # Create A2A message
                message = Message(
                    messageId=correlation_id,  # Use same ID for correlation
                    role=Role.USER,
                    parts=[TextPart(type="text", text=text)],
                )

                # Create SendMessageRequest with same correlation ID
                request = SendMessageRequest(
                    params=MessageSendParams(message=message),
                    id=correlation_id,  # Same ID for request/response matching
                )

                span.set_attribute("message_created", True)

                # Send message via A2A SDK
                with tracer.start_as_current_span("a2a_sdk_call") as sdk_span:
                    sdk_span.set_attributes(
                        {
                            "operation": "send_message",
                            "message_parts": len(message.parts),
                            "request_id": request.id,
                            "message_id": message.messageId,
                            "correlation_id": correlation_id,
                        }
                    )

                    response = await self.client.send_message(request)

                    sdk_span.set_attribute("response_received", True)

                # Extract response text
                response_text = ""
                if hasattr(response, "root") and hasattr(response.root, "result"):
                    result = response.root.result
                    if hasattr(result, "parts"):
                        for part in result.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                response_text += part.root.text
                    elif hasattr(result, "status") and hasattr(
                        result.status, "message"
                    ):
                        # Task response
                        if result.status.message and hasattr(
                            result.status.message, "parts"
                        ):
                            for part in result.status.message.parts:
                                if hasattr(part, "root") and hasattr(part.root, "text"):
                                    response_text += part.root.text

                # Log agent response with correlation
                self._log_conversation_item(
                    "agent_response", response_text, {"correlation_id": correlation_id}
                )

                span.set_attributes(
                    {
                        "response_received": True,
                        "response_text": response_text[:200],  # Truncate for logging
                        "response_length": len(response_text),
                        "output.value": response_text,
                        "correlation_id": correlation_id,
                    }
                )

                return response_text

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", str(e))
                self._log_conversation_item(
                    "error", f"Error: {str(e)}", {"correlation_id": correlation_id}
                )
                raise e

    async def send_streaming_message(self, text: str):
        """Send a streaming message to the A2A server and yield responses as they arrive"""

        # Log user message
        self._log_conversation_item("user_message", text)

        with tracer.start_as_current_span("a2a_client_send_streaming_message") as span:
            # Generate a single correlation ID for this streaming message exchange
            correlation_id = str(uuid.uuid4())

            span.set_attributes(
                {
                    "server_url": self.server_url,
                    "session_id": self.session_id,
                    "correlation_id": correlation_id,
                    "message_text": text[:200],
                    "message_length": len(text),
                    "input.value": text,
                    "session.id": self.session_id,
                    "streaming": True,
                }
            )

            try:
                # Create A2A message
                message = Message(
                    messageId=correlation_id,  # Use same ID for correlation
                    role=Role.USER,
                    parts=[TextPart(type="text", text=text)],
                )

                # Create SendStreamingMessageRequest with same correlation ID
                request = SendStreamingMessageRequest(
                    params=MessageSendParams(message=message),
                    id=correlation_id,  # Same ID for request/response matching
                )

                span.set_attribute("streaming_request_created", True)

                # Send streaming message via A2A SDK
                with tracer.start_as_current_span("a2a_streaming_sdk_call") as sdk_span:
                    sdk_span.set_attributes(
                        {
                            "operation": "send_streaming_message",
                            "message_parts": len(message.parts),
                            "request_id": request.id,
                            "message_id": message.messageId,
                            "correlation_id": correlation_id,
                        }
                    )

                    complete_response = ""
                    chunk_count = 0

                    async for response in self.client.send_message_streaming(request):
                        chunk_count += 1

                        # Process the streaming response
                        if hasattr(response, "root") and hasattr(
                            response.root, "result"
                        ):
                            result = response.root.result

                            # Handle different types of streaming responses
                            if hasattr(result, "kind"):
                                if result.kind == "message":
                                    # Message chunk
                                    chunk_text = ""
                                    if hasattr(result, "parts"):
                                        for part in result.parts:
                                            if hasattr(part, "root") and hasattr(
                                                part.root, "text"
                                            ):
                                                chunk_text += part.root.text

                                    if chunk_text:
                                        self._log_conversation_item(
                                            "stream_chunk",
                                            chunk_text,
                                            {
                                                "correlation_id": correlation_id,
                                                "chunk_number": chunk_count,
                                            },
                                        )
                                        complete_response += chunk_text
                                        yield chunk_text

                                elif result.kind == "status-update":
                                    # Status update
                                    status_msg = f"Status: {result.status.state}"
                                    if result.status.message:
                                        status_text = ""
                                        if hasattr(result.status.message, "parts"):
                                            for part in result.status.message.parts:
                                                if hasattr(part, "root") and hasattr(
                                                    part.root, "text"
                                                ):
                                                    status_text += part.root.text
                                        if status_text:
                                            status_msg += f" - {status_text}"

                                    self._log_conversation_item(
                                        "status_update",
                                        status_msg,
                                        {"correlation_id": correlation_id},
                                    )
                                    yield f"[{status_msg}]"

                                elif result.kind == "artifact-update":
                                    # Artifact update
                                    artifact_msg = (
                                        f"Artifact: {result.artifact.name or 'Unnamed'}"
                                    )
                                    if result.artifact.description:
                                        artifact_msg += (
                                            f" - {result.artifact.description}"
                                        )

                                    self._log_conversation_item(
                                        "artifact",
                                        artifact_msg,
                                        {"correlation_id": correlation_id},
                                    )
                                    yield f"[{artifact_msg}]"

                    # Log complete response with correlation
                    if complete_response:
                        self._log_conversation_item(
                            "agent_response",
                            complete_response,
                            {
                                "correlation_id": correlation_id,
                                "total_chunks": chunk_count,
                            },
                        )

                    span.set_attributes(
                        {
                            "streaming_complete": True,
                            "total_chunks": chunk_count,
                            "complete_response": complete_response[:200],
                            "correlation_id": correlation_id,
                        }
                    )

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", str(e))
                self._log_conversation_item(
                    "error",
                    f"Streaming error: {str(e)}",
                    {"correlation_id": correlation_id},
                )
                raise e

    def get_conversation_log(self) -> List[Dict[str, Any]]:
        """Get the complete conversation log"""
        return self.conversation_log.copy()

    def print_conversation_summary(self):
        """Print a summary of the conversation"""
        print("\n📊 Conversation Summary")
        print("=" * 50)

        user_messages = len(
            [item for item in self.conversation_log if item["type"] == "user_message"]
        )
        agent_responses = len(
            [item for item in self.conversation_log if item["type"] == "agent_response"]
        )
        status_updates = len(
            [item for item in self.conversation_log if item["type"] == "status_update"]
        )
        artifacts = len(
            [item for item in self.conversation_log if item["type"] == "artifact"]
        )
        stream_chunks = len(
            [item for item in self.conversation_log if item["type"] == "stream_chunk"]
        )
        errors = len(
            [item for item in self.conversation_log if item["type"] == "error"]
        )

        print(f"👤 User Messages: {user_messages}")
        print(f"🤖 Agent Responses: {agent_responses}")
        print(f"⚡ Stream Chunks: {stream_chunks}")
        print(f"📊 Status Updates: {status_updates}")
        print(f"📎 Artifacts: {artifacts}")
        print(f"❌ Errors: {errors}")
        print(f"🆔 Session ID: {self.session_id}")

        if self.conversation_log:
            first_msg = self.conversation_log[0]["timestamp"]
            last_msg = self.conversation_log[-1]["timestamp"]
            print(f"⏰ Duration: {first_msg} → {last_msg}")

    async def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card from the A2A server"""

        with tracer.start_as_current_span("a2a_client_get_agent_card") as span:
            span.set_attribute("server_url", self.server_url)

            try:
                # Get agent card via A2A SDK card resolver
                card = await self.card_resolver.get_agent_card()

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

    async def close(self):
        """Close the httpx client"""
        await self.httpx_client.aclose()


async def interactive_demo():
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

        # Get agent card
        print("\n📋 Getting agent information...")
        agent_card = await client.get_agent_card()
        print(f"✅ Connected to: {agent_card.get('name', 'Unknown Agent')}")
        print(f"📝 Description: {agent_card.get('description', 'No description')}")
        print(f"🔧 Skills available: {len(agent_card.get('skills', []))}")

        # Interactive chat loop
        print("\n💬 Interactive Chat")
        print("💡 Commands:")
        print("  • Type your message for regular response")
        print("  • Type 'stream: <message>' for streaming response")
        print("  • Type 'summary' to see conversation summary")
        print("  • Type 'quit' to exit")
        print("💡 Try: 'What's the weather in Tokyo?' or 'stream: Calculate 15 * 7'")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "summary":
                    client.print_conversation_summary()
                    continue

                if not user_input:
                    continue

                # Check if user wants streaming
                if user_input.lower().startswith("stream:"):
                    message = user_input[7:].strip()
                    if message:
                        print("🤖 Agent (streaming): ", end="", flush=True)

                        # Send streaming message
                        async for chunk in client.send_streaming_message(message):
                            print(chunk, end="", flush=True)
                        print()  # New line after streaming is complete
                else:
                    print("🤖 Agent: ", end="", flush=True)

                    # Send regular message
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
    finally:
        if client:
            client.print_conversation_summary()
            await client.close()


async def streaming_demo():
    """Demo specifically for streaming capabilities"""

    print("🌊 A2A Streaming Demo with Phoenix Observability")
    print("=" * 50)

    client = None
    try:
        client = ObservableA2AClient("http://localhost:8000")

        # Test streaming messages
        streaming_messages = [
            "Tell me a story about a robot learning to dance",
            "What's the weather in multiple cities: New York, Tokyo, and London?",
            "Calculate these math problems step by step: 25 * 4, then 100 / 5, then add them together",
            "Explain how AI agents work in simple terms",
        ]

        print(f"\n🧪 Running {len(streaming_messages)} streaming interactions...")

        for i, message in enumerate(streaming_messages, 1):
            print(f"\n📤 Streaming Test {i}: {message}")
            print("🌊 Response: ", end="", flush=True)

            try:
                async for chunk in client.send_streaming_message(message):
                    print(chunk, end="", flush=True)
                print()  # New line after each complete response

                # Small delay between requests
                await asyncio.sleep(2)

            except Exception as e:
                print(f"\n❌ Error in streaming test {i}: {e}")

        print("\n✅ Streaming demo completed!")
        print("📊 Check Phoenix UI at http://127.0.0.1:6006 for traces and metrics")

    except Exception as e:
        print(f"❌ Streaming demo failed: {e}")
    finally:
        if client:
            client.print_conversation_summary()
            await client.close()


async def automated_demo():
    """Automated demo showing various A2A interactions"""

    print("🤖 Automated A2A Demo with Phoenix Observability")
    print("=" * 50)

    client = None
    try:
        client = ObservableA2AClient("http://localhost:8000")

        # Test messages (mix of regular and streaming)
        test_scenarios = [
            {"message": "Hello! Can you introduce yourself?", "streaming": False},
            {"message": "What's the weather like in New York?", "streaming": False},
            {"message": "Can you calculate 25 * 4?", "streaming": True},
            {"message": "What's the weather in Tokyo and London?", "streaming": True},
            {
                "message": "Calculate the result of 100 divided by 5, then multiply by 3",
                "streaming": False,
            },
        ]

        print(f"\n🧪 Running {len(test_scenarios)} test interactions...")

        for i, scenario in enumerate(test_scenarios, 1):
            message = scenario["message"]
            is_streaming = scenario["streaming"]
            mode = "STREAMING" if is_streaming else "REGULAR"

            print(f"\n📤 Test {i} ({mode}): {message}")

            try:
                if is_streaming:
                    print("🌊 Response: ", end="", flush=True)
                    async for chunk in client.send_streaming_message(message):
                        print(chunk, end="", flush=True)
                    print()  # New line
                else:
                    response = await client.send_message(message)
                    print(f"📥 Response: {response}")

                # Small delay between requests
                await asyncio.sleep(1)

            except Exception as e:
                print(f"❌ Error in test {i}: {e}")

        print("\n✅ Automated demo completed!")
        print("📊 Check Phoenix UI at http://127.0.0.1:6006 for traces and metrics")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        if client:
            client.print_conversation_summary()
            await client.close()


async def discovery_demo():
    """Demo of agent discovery capabilities"""

    print("🔍 A2A Agent Discovery Demo")
    print("=" * 50)

    # Discovery options
    print("\n🎯 Discovery Options:")
    print("1. Localhost discovery (scan local ports)")
    print("2. Local network discovery (scan network)")
    print("3. Auto-select first agent")
    print("4. Discovery with filtering")

    try:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            # Localhost discovery
            clients = await ObservableA2AClient.discover(discovery_scope="localhost")

            if clients:
                # Use the first discovered client
                client = clients[0]
                await demo_with_client(client, "localhost discovery")
                await client.close()

        elif choice == "2":
            # Local network discovery
            print("⚠️  This may take a while (scanning network)...")
            clients = await ObservableA2AClient.discover(
                discovery_scope="local-network"
            )

            if clients and len(clients) > 0:
                # Let user select
                selected = await ObservableA2AClient.discover_and_select(
                    discovery_scope="localhost"
                )
                if selected:
                    await demo_with_client(selected, "network discovery")
                    await selected.close()

        elif choice == "3":
            # Auto-select first agent
            client = await ObservableA2AClient.discover_and_select(
                discovery_scope="localhost", auto_select=True
            )

            if client:
                await demo_with_client(client, "auto-selected agent")
                await client.close()

        elif choice == "4":
            # Discovery with filtering
            print("\n🔍 Discovering agents with weather capabilities...")

            # Filter for agents with weather skills
            agent_filter = {
                "capabilities": ["streaming"],  # Agents that support streaming
                # "skills": ["get_weather"]     # Uncomment to filter by specific skills
            }

            clients = await ObservableA2AClient.discover(
                discovery_scope="localhost", agent_filter=agent_filter
            )

            if clients:
                client = clients[0]
                print(f"🎯 Found agent with streaming: {client.agent_card.name}")
                await demo_with_client(client, "filtered discovery")
                await client.close()
            else:
                print("❌ No agents found matching filter criteria")

        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n👋 Discovery demo interrupted")


async def demo_with_client(client: ObservableA2AClient, discovery_method: str):
    """Run a quick demo with a discovered client"""

    print(f"\n🎮 Quick demo with agent discovered via {discovery_method}")
    print("-" * 50)

    # Test messages
    test_messages = [
        "Hello! Can you introduce yourself?",
        "What's the weather in Tokyo?",
        "Calculate 15 * 7",
    ]

    for message in test_messages:
        print(f"\n📤 Sending: {message}")
        try:
            response = await client.send_message(message)
            print(f"📥 Response: {response}")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Error: {e}")

    # Show conversation summary
    client.print_conversation_summary()


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
    print(f"  • Phoenix: ✅ Enabled (UI at http://127.0.0.1:6006)")
    print("  • HTTP Tracing: ✅ Enabled")
    print("  • Conversation Logging: ✅ Enabled")
    print("  • Agent Discovery: ✅ Enabled")

    # Choose demo mode
    print("\n🎯 Choose demo mode:")
    print("1. Interactive chat (regular + streaming)")
    print("2. Streaming demo")
    print("3. Automated demo (mixed scenarios)")
    print("4. Agent discovery demo")

    try:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            await interactive_demo()
        elif choice == "2":
            await streaming_demo()
        elif choice == "3":
            await automated_demo()
        elif choice == "4":
            await discovery_demo()
        else:
            print("❌ Invalid choice. Running discovery demo...")
            await discovery_demo()

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
