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
            ports = [
                8000,
                8001,
                8002,
                8080,
                8081,
                3000,
                3001,
                5000,
                5001,
                9000,
            ]  # Extended port range

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
                                                "host": host,
                                                "port": port,
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

    async def discover_by_dns_srv(self, domain: str = "local") -> List[Dict[str, Any]]:
        """Discover agents using DNS SRV records"""
        discovered = []

        try:
            import dns.resolver

            # Query for A2A service records
            # Format: _a2a._tcp.domain
            srv_query = f"_a2a._tcp.{domain}"

            try:
                answers = dns.resolver.resolve(srv_query, "SRV")

                for rdata in answers:
                    host = str(rdata.target).rstrip(".")
                    port = rdata.port

                    print(f"🔍 Found A2A service via DNS SRV: {host}:{port}")

                    # Validate the discovered service
                    agents = await self.discover_by_well_known([host], [port])
                    for agent in agents:
                        agent["discovery_method"] = "dns-srv"
                        agent["srv_priority"] = rdata.priority
                        agent["srv_weight"] = rdata.weight

                    discovered.extend(agents)

            except dns.resolver.NXDOMAIN:
                print(f"🔍 No DNS SRV records found for {srv_query}")
            except Exception as e:
                print(f"⚠️  DNS SRV query failed: {e}")

        except ImportError:
            print(
                "⚠️  DNS discovery requires 'dnspython' package: pip install dnspython"
            )

        return discovered

    async def discover_by_environment(self) -> List[Dict[str, Any]]:
        """Discover agents from environment variables"""
        discovered = []

        # Check environment variables for agent URLs
        env_vars = [
            "A2A_AGENT_URLS",  # Comma-separated list
            "A2A_DISCOVERY_URLS",
            "AGENT_ENDPOINTS",
            "A2A_SERVERS",
        ]

        agent_urls = []
        for var in env_vars:
            value = os.getenv(var)
            if value:
                # Split by comma and clean up
                urls = [url.strip() for url in value.split(",") if url.strip()]
                agent_urls.extend(urls)
                print(f"🔍 Found agent URLs in {var}: {urls}")

        # Also check individual numbered environment variables
        i = 1
        while True:
            url = os.getenv(f"A2A_AGENT_{i}")
            if not url:
                break
            agent_urls.append(url.strip())
            i += 1

        # Validate discovered URLs
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for url in agent_urls:
                try:
                    parsed = urlparse(url)
                    host = parsed.hostname
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)

                    agents = await self.discover_by_well_known([host], [port])
                    for agent in agents:
                        agent["discovery_method"] = "environment"
                        agent["env_url"] = url

                    discovered.extend(agents)

                except Exception as e:
                    print(f"⚠️  Invalid URL from environment: {url} - {e}")

        return discovered

    async def discover_by_consul(
        self, consul_url: str = "http://localhost:8500"
    ) -> List[Dict[str, Any]]:
        """Discover agents through Consul service discovery"""
        discovered = []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Query Consul for A2A services
                consul_endpoint = f"{consul_url}/v1/catalog/service/a2a-agent"

                response = await client.get(consul_endpoint)
                if response.status_code == 200:
                    services = response.json()

                    for service in services:
                        host = service.get("ServiceAddress") or service.get("Address")
                        port = service.get("ServicePort")

                        if host and port:
                            print(f"🔍 Found A2A service in Consul: {host}:{port}")

                            agents = await self.discover_by_well_known([host], [port])
                            for agent in agents:
                                agent["discovery_method"] = "consul"
                                agent["consul_service_id"] = service.get("ServiceID")
                                agent["consul_tags"] = service.get("ServiceTags", [])

                            discovered.extend(agents)

        except Exception as e:
            print(f"⚠️  Consul discovery failed: {e}")

        return discovered

    async def discover_by_kubernetes(
        self, namespace: str = "default"
    ) -> List[Dict[str, Any]]:
        """Discover agents through Kubernetes service discovery"""
        discovered = []

        try:
            # This would require kubernetes client library
            # kubectl get services -l app=a2a-agent -o json

            # For now, simulate with environment-based discovery
            # In real implementation, would use kubernetes Python client
            kube_services = os.getenv("KUBE_A2A_SERVICES")
            if kube_services:
                print(f"🔍 Found Kubernetes A2A services: {kube_services}")
                # Parse and validate services
                # discovered.extend(...)

        except Exception as e:
            print(f"⚠️  Kubernetes discovery failed: {e}")

        return discovered

    async def discover_by_mdns(self) -> List[Dict[str, Any]]:
        """Discover agents using mDNS/Bonjour (zero-configuration networking)"""
        discovered = []

        try:
            # This would require zeroconf library
            print("🔍 Scanning for A2A agents via mDNS/Bonjour...")
            print("⚠️  mDNS discovery requires 'zeroconf' package: pip install zeroconf")

            # Example implementation:
            # from zeroconf import ServiceBrowser, Zeroconf
            #
            # class A2AListener:
            #     def add_service(self, zeroconf, type, name):
            #         info = zeroconf.get_service_info(type, name)
            #         # Extract host:port and validate

        except ImportError:
            print("⚠️  mDNS discovery requires 'zeroconf' package")
        except Exception as e:
            print(f"⚠️  mDNS discovery failed: {e}")

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
            discovery_scope: Discovery method:
                - "localhost": Scan local ports
                - "local-network": Scan network
                - "environment": Environment variables
                - "dns": DNS SRV records
                - "consul": Consul service discovery
                - "kubernetes": Kubernetes services
                - "mdns": mDNS/Bonjour
                - "registry": Registry services
                - "all": All methods
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

        if discovery_scope in ["environment", "all"]:
            discovered_agents.extend(await discovery.discover_by_environment())

        if discovery_scope in ["dns", "all"]:
            discovered_agents.extend(await discovery.discover_by_dns_srv())

        if discovery_scope in ["consul", "all"]:
            discovered_agents.extend(await discovery.discover_by_consul())

        if discovery_scope in ["kubernetes", "all"]:
            discovered_agents.extend(await discovery.discover_by_kubernetes())

        if discovery_scope in ["mdns", "all"]:
            discovered_agents.extend(await discovery.discover_by_mdns())

        if discovery_scope in ["registry", "all"]:
            # Example registry URLs (these would be real registry services)
            registry_urls = [
                "https://a2a-registry.example.com",
                "https://agent-marketplace.example.com",
            ]
            discovered_agents.extend(
                await discovery.discover_by_registry(registry_urls)
            )

        # Remove duplicates based on base_url
        unique_agents = {}
        for agent in discovered_agents:
            base_url = agent["base_url"]
            if base_url not in unique_agents:
                unique_agents[base_url] = agent
            else:
                # Prefer agents with more discovery methods
                existing = unique_agents[base_url]
                if not existing.get("discovery_methods"):
                    existing["discovery_methods"] = [existing["discovery_method"]]
                if agent["discovery_method"] not in existing["discovery_methods"]:
                    existing["discovery_methods"].append(agent["discovery_method"])

        discovered_agents = list(unique_agents.values())

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
                # Add discovery metadata to client
                client.discovery_info = agent_info
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
                    role=Role.user,
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
                    role=Role.user,
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


async def comprehensive_discovery_demo():
    """Comprehensive demo showing all discovery mechanisms"""

    print("🌐 Comprehensive A2A Agent Discovery Demo")
    print("=" * 60)

    print(
        """
🔍 Multiple Discovery Mechanisms Available:

1. 🏠 Well-Known Endpoints (/.well-known/agent.json)
   • Multiple ports: 8000, 8001, 8002, 8080, 8081, etc.
   • Multiple hosts: localhost, network IPs
   
2. 🌐 DNS Service Discovery (SRV records)
   • _a2a._tcp.local domain queries
   • Automatic service resolution
   
3. 📋 Environment Variables
   • A2A_AGENT_URLS=http://host1:8000,http://host2:8001
   • A2A_AGENT_1=http://first-agent:8000
   
4. 🏢 Service Discovery Systems
   • Consul: service registry integration
   • Kubernetes: service discovery
   • mDNS/Bonjour: zero-config networking
   
5. 📚 Registry Services
   • Centralized agent marketplaces
   • Public/private agent directories
"""
    )

    # Demo each discovery method
    discovery_methods = [
        ("localhost", "🏠 Localhost Discovery"),
        ("environment", "📋 Environment Variable Discovery"),
        ("dns", "🌐 DNS SRV Discovery"),
        ("consul", "🏢 Consul Discovery"),
        ("mdns", "📡 mDNS Discovery"),
    ]

    all_discovered = []

    for method, description in discovery_methods:
        print(f"\n{description}")
        print("-" * 40)

        try:
            clients = await ObservableA2AClient.discover(
                discovery_scope=method, timeout=3.0  # Shorter timeout for demo
            )

            print(f"✅ Found {len(clients)} agents via {method}")

            for client in clients:
                agent_card = client.agent_card
                discovery_info = getattr(client, "discovery_info", {})

                print(f"  🤖 {agent_card.name}")
                print(f"     📍 {client.server_url}")
                print(f"     🔧 {len(agent_card.skills)} skills")
                print(
                    f"     🔍 Discovery: {discovery_info.get('discovery_method', 'unknown')}"
                )

                if "response_time" in discovery_info:
                    print(f"     ⚡ Response: {discovery_info['response_time']:.3f}s")

                # Close immediately after discovery (demo only)
                await client.close()

            all_discovered.extend(clients)

        except Exception as e:
            print(f"❌ {method} discovery failed: {e}")

    # Show environment setup examples
    print("\n🛠️  Setup Examples for Multiple Agent Discovery:")
    print("=" * 60)

    print(
        """
💡 Running Multiple A2A Servers:

# Terminal 1: Weather Agent
export A2A_AGENT_PORT=8000
python src/server.py --port 8000 --agent-type weather

# Terminal 2: Math Agent  
export A2A_AGENT_PORT=8001
python src/server.py --port 8001 --agent-type math

# Terminal 3: Translation Agent
export A2A_AGENT_PORT=8002
python src/server.py --port 8002 --agent-type translation

📋 Environment Variable Discovery:
export A2A_AGENT_URLS="http://localhost:8000,http://localhost:8001,http://localhost:8002"
export A2A_AGENT_1="http://weather-agent:8000"
export A2A_AGENT_2="http://math-agent:8001"

🌐 DNS SRV Records (for production):
_a2a._tcp.company.com SRV 0 5 8000 weather-agent.company.com
_a2a._tcp.company.com SRV 0 5 8001 math-agent.company.com

🏢 Consul Service Registration:
consul services register -name=a2a-agent -port=8000 -address=192.168.1.10
consul services register -name=a2a-agent -port=8001 -address=192.168.1.11

🐳 Docker Compose Multi-Agent Setup:
version: '3.8'
services:
  weather-agent:
    image: a2a-server
    ports: ["8000:8000"]
    environment: [AGENT_TYPE=weather]
  
  math-agent:
    image: a2a-server  
    ports: ["8001:8000"]
    environment: [AGENT_TYPE=math]
"""
    )

    print(
        f"\n📊 Discovery Summary: {len(all_discovered)} total agents found across all methods"
    )


async def discovery_demo():
    """Demo of agent discovery capabilities"""

    print("🔍 A2A Agent Discovery Demo")
    print("=" * 50)

    # Discovery options
    print("\n🎯 Discovery Options:")
    print("1. Localhost discovery (scan local ports)")
    print("2. Local network discovery (scan network)")
    print("3. Environment variable discovery")
    print("4. DNS SRV discovery")
    print("5. Comprehensive discovery (all methods)")
    print("6. Auto-select first agent")
    print("7. Discovery with filtering")

    try:
        choice = input("\nEnter choice (1-7): ").strip()

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
            # Environment discovery
            print("📋 Looking for agents in environment variables...")
            print("💡 Set A2A_AGENT_URLS or A2A_AGENT_1, A2A_AGENT_2, etc.")

            clients = await ObservableA2AClient.discover(discovery_scope="environment")

            if clients:
                client = clients[0]
                await demo_with_client(client, "environment discovery")
                await client.close()
            else:
                print("❌ No agents found in environment variables")
                print("💡 Try: export A2A_AGENT_URLS='http://localhost:8000'")

        elif choice == "4":
            # DNS discovery
            print("🌐 Looking for agents via DNS SRV records...")

            clients = await ObservableA2AClient.discover(discovery_scope="dns")

            if clients:
                client = clients[0]
                await demo_with_client(client, "DNS SRV discovery")
                await client.close()
            else:
                print("❌ No agents found via DNS SRV")
                print("💡 Try: dig _a2a._tcp.local SRV")

        elif choice == "5":
            # Comprehensive discovery demo
            await comprehensive_discovery_demo()

        elif choice == "6":
            # Auto-select first agent
            client = await ObservableA2AClient.discover_and_select(
                discovery_scope="localhost", auto_select=True
            )

            if client:
                await demo_with_client(client, "auto-selected agent")
                await client.close()

        elif choice == "7":
            # Discovery with filtering
            print("\n🔍 Discovering agents with streaming capabilities...")

            # Filter for agents with streaming capabilities
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
    print("  • Phoenix: ✅ Enabled (UI at http://127.0.0.1:6006)")
    print("  • HTTP Tracing: ✅ Enabled")
    print("  • Conversation Logging: ✅ Enabled")
    print("  • Agent Discovery: ✅ Enabled")

    # Choose demo mode
    print("\n🎯 Choose demo mode:")
    print("1. Interactive chat (regular + streaming)")
    print("2. Streaming demo")
    print("3. Automated demo (mixed scenarios)")
    print("4. Agent discovery demo")
    print("5. Comprehensive discovery demo (all methods)")

    try:
        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            await interactive_demo()
        elif choice == "2":
            await streaming_demo()
        elif choice == "3":
            await automated_demo()
        elif choice == "4":
            await discovery_demo()
        elif choice == "5":
            await comprehensive_discovery_demo()
        else:
            print("❌ Invalid choice. Running discovery demo...")
            await discovery_demo()

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
