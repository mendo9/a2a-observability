# A2A Server with OpenAI Agent and Phoenix Observability

This project implements a Google A2A (Agent-to-Agent) server with an OpenAI agent that has weather and calculation capabilities, fully instrumented with [Phoenix](https://github.com/Arize-ai/phoenix) for comprehensive AI observability.

## Features

- 🤖 **OpenAI Agent**: GPT-4o-mini powered agent with function calling
- 🌤️ **Weather Tool**: Get weather information for any location
- 🧮 **Calculator Tool**: Perform basic mathematical operations
- 📊 **Phoenix Observability**: Complete tracing of AI interactions, tool usage, and performance metrics
- 🔗 **A2A Protocol**: Google's Agent-to-Agent communication standard
- 📈 **OpenTelemetry**: Industry-standard observability

## Prerequisites

- Python 3.8+
- OpenAI API key
- A2A SDK (contact Google for access)
- OpenAI Agents SDK (contact OpenAI for access)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd a2a-observability
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Phoenix**
   ```bash
   pip install phoenix openinference-instrumentation-openai
   ```

## Configuration

### Required Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Optional Environment Variables

```bash
# Phoenix configuration (defaults to local)
export PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:6006/v1/traces"

# Logfire (optional)
export LOGFIRE_TOKEN="your-logfire-token"

# External OTLP endpoint (optional)
export OTEL_EXPORTER_OTLP_ENDPOINT="your-otlp-endpoint"
export OTEL_EXPORTER_OTLP_HEADERS="your-auth-headers"
```

## Running the Server

1. **Start Phoenix (if not already running)**
   ```bash
   python -m phoenix.server.main serve
   ```
   This will start Phoenix UI at http://127.0.0.1:6006

2. **Start the A2A server**
   ```bash
   python src/server.py
   ```

The server will start on `http://localhost:8000` with the following endpoints:
- **A2A RPC**: `http://localhost:8000/` (JSON-RPC for messages)
- **Agent Card**: `http://localhost:8000/.well-known/agent.json` (agent discovery)
- **Phoenix UI**: `http://127.0.0.1:6006` (observability dashboard)

## A2A Protocol Endpoints

The Google A2A protocol defines specific endpoint conventions:

### Standard Endpoints
| Endpoint | Purpose | Method |
|----------|---------|---------|
| `/` | JSON-RPC endpoint for all A2A operations | POST |
| `/.well-known/agent.json` | Agent discovery and capabilities | GET |
| `/agent/authenticatedExtendedCard` | Extended agent info (if auth supported) | GET |

### Why Root Path for RPC?
- **🏛️ Protocol Standard**: A2A specification uses root `/` for JSON-RPC
- **🔄 Single Endpoint**: All message types routed through one endpoint
- **📡 JSON-RPC Convention**: Standard practice for JSON-RPC APIs
- **🎯 Simplified Routing**: Agent card for discovery, root for operations

## Usage Examples

### Weather Queries
- "What's the weather in Tokyo?"
- "Tell me the weather in New York"
- "How's the weather in London?"

### Math Calculations
- "Calculate 15 * 7"
- "What's 100 divided by 5?"
- "Add 25 and 37"

## Observability Features

### Phoenix Dashboard
Access the Phoenix UI at http://127.0.0.1:6006 to see:
- **Traces**: Complete execution traces of agent interactions
- **Spans**: Individual operations (LLM calls, tool usage, etc.)
- **Performance Metrics**: Latency, token usage, error rates
- **Tool Usage**: Function calling patterns and results
- **Session Tracking**: User conversation flows

### Traced Operations
- ✅ OpenAI LLM calls (automatically instrumented)
- ✅ Function tool executions
- ✅ A2A protocol interactions
- ✅ Error handling and exceptions
- ✅ Session and conversation tracking

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   A2A Client    │────▶│  A2A Server  │────▶│ OpenAI API  │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
                       ┌──────────────┐    ┌─────────────┐
                       │    Tools     │    │   Phoenix   │
                       │ • Weather    │    │   Tracing   │
                       │ • Calculator │    │     UI      │
                       └──────────────┘    └─────────────┘
```

## Agent Capabilities

The agent supports:
- **Streaming**: Real-time response streaming
- **Function Calling**: Weather and calculation tools
- **State Tracking**: Conversation history and context
- **Error Handling**: Graceful error recovery
- **Observability**: Complete execution tracing

## Troubleshooting

### Common Issues

1. **Phoenix not connecting**
   - Ensure Phoenix server is running on port 6006
   - Check PHOENIX_COLLECTOR_ENDPOINT environment variable

2. **OpenAI API errors**
   - Verify OPENAI_API_KEY is set correctly
   - Check API key permissions and quotas

3. **Missing dependencies**
   - Ensure A2A SDK and OpenAI Agents SDK are properly installed
   - Check that all requirements.txt packages are installed

### Debug Mode
Enable console span export by checking the console output when running the server. All spans will be printed to the console for debugging.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add appropriate tracing for new features
5. Test with Phoenix observability
6. Submit a pull request

## License

[Add your license here]

## Links

- [Phoenix Documentation](https://docs.arize.com/phoenix)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Google A2A Protocol](https://developers.google.com/agent-to-agent)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

## ID Strategy and Correlation

### ID Types and Their Purposes

The A2A client uses a strategic approach to ID generation for optimal tracing and correlation:

#### 🔗 **Correlation ID** (Single UUID per exchange)
```python
correlation_id = str(uuid.uuid4())  # Generated once per message exchange

# Used for:
message.messageId = correlation_id      # Message content identifier
request.id = correlation_id             # JSON-RPC request identifier
```

**Why the same ID?**
- ✅ **Easy correlation**: Links request, message, and response
- ✅ **Simplified tracing**: Single ID tracks entire message flow
- ✅ **Debugging**: Easy to find related events in logs
- ✅ **Phoenix visibility**: All related spans grouped together

#### 🌐 **Session ID** (Persistent across conversation)
```python
self.session_id = str(uuid.uuid4())  # Generated once per client instance
```

**Purpose:**
- Groups all messages in a conversation
- Enables conversation-level analytics
- Tracks user sessions across multiple exchanges

### ID Usage Patterns

| ID Type | Scope | When Generated | Used For |
|---------|-------|----------------|----------|
| **Correlation ID** | Single message exchange | Per send_message() call | Request/Response matching, Message identification |
| **Session ID** | Entire conversation | Client initialization | Conversation grouping, Session analytics |

### Example ID Flow

```
Session: abc-123-def
├── Message Exchange 1: msg-456-ghi
│   ├── Request ID: msg-456-ghi
│   ├── Message ID: msg-456-ghi
│   └── Response: linked to msg-456-ghi
└── Message Exchange 2: msg-789-jkl
    ├── Request ID: msg-789-jkl
    ├── Message ID: msg-789-jkl
    └── Response: linked to msg-789-jkl
```

### Benefits of This Approach

1. **🔍 Tracing**: Phoenix can group all spans by correlation_id
2. **📊 Analytics**: Session-level conversation metrics
3. **🐛 Debugging**: Single ID finds entire message flow
4. **🔗 Correlation**: Request/response/message all linked
5. **📈 Observability**: Clear parent-child relationships

## Multiple A2A Server Discovery

### 🌐 **Multiple Discovery Mechanisms**

A2A agents can be discovered through **multiple mechanisms**, not just a single well-known address. The `/.well-known/agent.json` endpoint is just one of many discovery methods:

#### **1. 🏠 Well-Known Endpoint Discovery**
Each A2A server exposes its agent card at `/.well-known/agent.json`:

```bash
# Multiple agents on same host (different ports)
http://localhost:8000/.well-known/agent.json  # Weather Agent
http://localhost:8001/.well-known/agent.json  # Math Agent
http://localhost:8002/.well-known/agent.json  # Translation Agent

# Multiple agents on different hosts
http://weather-server:8000/.well-known/agent.json
http://math-server:8000/.well-known/agent.json  
http://translate-server:8000/.well-known/agent.json
```

#### **2. 📋 Environment Variable Discovery**
Configure agent URLs through environment variables:

```bash
# Comma-separated list
export A2A_AGENT_URLS="http://localhost:8000,http://localhost:8001,http://localhost:8002"

# Individual numbered variables
export A2A_AGENT_1="http://weather-agent:8000"
export A2A_AGENT_2="http://math-agent:8001"
export A2A_AGENT_3="http://translate-agent:8002"

# Alternative variable names
export AGENT_ENDPOINTS="http://host1:8000,http://host2:8001"
export A2A_SERVERS="http://server1:8000,http://server2:8000"
```

#### **3. 🌐 DNS Service Discovery (SRV Records)**
Use DNS SRV records for scalable discovery:

```bash
# DNS SRV record format: _a2a._tcp.domain
_a2a._tcp.company.com SRV 0 5 8000 weather-agent.company.com
_a2a._tcp.company.com SRV 0 5 8001 math-agent.company.com
_a2a._tcp.company.com SRV 0 5 8002 translate-agent.company.com

# Query SRV records
dig _a2a._tcp.company.com SRV
```

#### **4. 🏢 Service Discovery Systems**

**Consul Service Discovery:**
```bash
# Register agents with Consul
consul services register -name=a2a-agent -port=8000 -address=192.168.1.10 -tag=weather
consul services register -name=a2a-agent -port=8001 -address=192.168.1.11 -tag=math

# Query Consul for A2A agents
curl http://localhost:8500/v1/catalog/service/a2a-agent
```

**Kubernetes Service Discovery:**
```yaml
# Kubernetes service definition
apiVersion: v1
kind: Service
metadata:
  name: weather-agent
  labels:
    app: a2a-agent
    type: weather
spec:
  selector:
    app: weather-agent
  ports:
  - port: 8000
    targetPort: 8000
```

**mDNS/Bonjour (Zero-Config):**
```bash
# Broadcast A2A services via mDNS
avahi-publish-service "Weather Agent" _a2a._tcp 8000
avahi-publish-service "Math Agent" _a2a._tcp 8001

# Browse for A2A services
avahi-browse _a2a._tcp
```

### 🎯 **Discovery Usage Examples**

#### **Basic Discovery**
```python
from src.client import ObservableA2AClient

# Discover agents on localhost
clients = await ObservableA2AClient.discover(discovery_scope="localhost")

# Discover via environment variables
clients = await ObservableA2AClient.discover(discovery_scope="environment")

# Discover via DNS SRV
clients = await ObservableA2AClient.discover(discovery_scope="dns")

# Discover via all methods
clients = await ObservableA2AClient.discover(discovery_scope="all")
```

#### **Filtered Discovery**
```python
# Find agents with specific capabilities
agent_filter = {
    "capabilities": ["streaming"],      # Must support streaming
    "skills": ["weather", "calculate"], # Must have weather or calculate skills
    "name_pattern": "openai"           # Name must contain "openai"
}

clients = await ObservableA2AClient.discover(
    discovery_scope="all",
    agent_filter=agent_filter
)
```

#### **Discovery Scopes**

| Scope | Description | Use Case |
|-------|-------------|----------|
| `"localhost"` | Scan localhost ports 8000-9000 | Local development |
| `"local-network"` | Scan local network subnet | LAN deployment |
| `"environment"` | Environment variable URLs | Configuration-based |
| `"dns"` | DNS SRV record queries | Production DNS setup |
| `"consul"` | Consul service registry | Microservices architecture |
| `"kubernetes"` | Kubernetes service discovery | Container orchestration |
| `"mdns"` | mDNS/Bonjour zero-config | Local network auto-discovery |
| `"registry"` | Public agent registries | Agent marketplaces |
| `"all"` | All discovery methods | Comprehensive search |

### 🚀 **Production Deployment Patterns**

#### **1. Multi-Port Single Host**
```bash
# Run multiple specialized agents on one server
docker run -p 8000:8000 a2a-server --agent-type weather
docker run -p 8001:8000 a2a-server --agent-type math  
docker run -p 8002:8000 a2a-server --agent-type translate
```

#### **2. Multi-Host Deployment**
```bash
# Distributed across multiple servers
Server 1 (192.168.1.10): Weather Agent on port 8000
Server 2 (192.168.1.11): Math Agent on port 8000
Server 3 (192.168.1.12): Translation Agent on port 8000
```

#### **3. Docker Compose Multi-Agent**
```yaml
version: '3.8'
services:
  weather-agent:
    image: a2a-server
    ports: ["8000:8000"]
    environment:
      - AGENT_TYPE=weather
      - PHOENIX_ENDPOINT=http://phoenix:6006
    
  math-agent:
    image: a2a-server
    ports: ["8001:8000"]
    environment:
      - AGENT_TYPE=math
      - PHOENIX_ENDPOINT=http://phoenix:6006
      
  translate-agent:
    image: a2a-server
    ports: ["8002:8000"]
    environment:
      - AGENT_TYPE=translate
      - PHOENIX_ENDPOINT=http://phoenix:6006

  phoenix:
    image: phoenix-server
    ports: ["6006:6006"]
```

#### **4. Kubernetes Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: a2a-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: a2a-agent
  template:
    metadata:
      labels:
        app: a2a-agent
    spec:
      containers:
      - name: weather-agent
        image: a2a-server
        ports:
        - containerPort: 8000
        env:
        - name: AGENT_TYPE
          value: "weather"
---
apiVersion: v1
kind: Service
metadata:
  name: a2a-agents
  labels:
    app: a2a-agent
spec:
  selector:
    app: a2a-agent
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### 🔍 **Discovery Algorithm**

The discovery process works as follows:

1. **🔍 Scan Phase**: Check all specified discovery sources
2. **✅ Validation Phase**: Fetch `/.well-known/agent.json` from each candidate
3. **🧪 Verification Phase**: Validate agent card structure using `AgentCard.model_validate()`
4. **🔧 Filtering Phase**: Apply user-defined criteria (skills, capabilities, etc.)
5. **🔗 Connection Phase**: Create connected `A2AClient` instances
6. **📊 Deduplication Phase**: Remove duplicate agents found via multiple methods

### 💡 **Best Practices**

**Development:**
- Use `localhost` discovery for local testing
- Set environment variables for known agents

**Production:**
- Use DNS SRV records for scalable discovery
- Implement service registry integration (Consul, etcd)
- Use load balancers for agent high availability

**Security:**
- Validate agent certificates in production
- Use authentication for sensitive agents
- Implement agent authorization and access control

### 🌐 **Why Multiple Discovery Methods?**

1. **🔄 Flexibility**: Different deployment scenarios need different discovery
2. **📈 Scalability**: DNS and service registries handle large agent populations  
3. **🛡️ Resilience**: Multiple discovery paths provide fallback options
4. **🏗️ Architecture**: Different architectures (monolith, microservices, serverless) need different approaches
5. **🌍 Environments**: Development, staging, production have different requirements

**The `/.well-known/agent.json` endpoint is just the foundation** - real-world A2A deployments use sophisticated discovery mechanisms to handle dozens or hundreds of specialized agents across distributed infrastructure.

## Agent Discovery

The A2A client supports automatic agent discovery through multiple mechanisms:

### 🔍 **Discovery Methods**

#### 1. **Manual Connection** (Traditional)
```python
client = ObservableA2AClient("http://localhost:8000")
```

#### 2. **Automatic Discovery**
```python
# Discover agents on localhost
clients = await ObservableA2AClient.discover(discovery_scope="localhost")

# Discover and auto-select first agent
client = await ObservableA2AClient.discover_and_select(auto_select=True)

# Discover agents on local network
clients = await ObservableA2AClient.discover(discovery_scope="local-network")
```

#### 3. **Filtered Discovery**
```python
# Find agents with specific capabilities
agent_filter = {
    "capabilities": ["streaming"],      # Must support streaming
    "skills": ["weather", "calculate"], # Must have weather or calculate skills
    "name_pattern": "openai"           # Name must contain "openai"
}

clients = await ObservableA2AClient.discover(
    discovery_scope="localhost",
    agent_filter=agent_filter
)
```

### 🌐 **Discovery Scopes**

| Scope | Description | Use Case |
|-------|-------------|----------|
| `"localhost"` | Scan common ports on localhost | Local development |
| `"local-network"` | Scan local network subnet | LAN deployment |
| `"registry"` | Query agent registries | Production discovery |
| `"all"` | All of the above | Comprehensive search |

### 🎯 **Discovery Process**

1. **Port Scanning**: Checks common ports (8000, 8080, 3000, 5000, 9000)
2. **Agent Card Validation**: Fetches `/.well-known/agent.json`
3. **Capability Verification**: Validates agent card structure
4. **Filtering**: Applies user-defined criteria
5. **Client Creation**: Instantiates connected clients

### 📡 **Well-Known Endpoint Discovery**

The discovery follows the A2A standard:

```http
GET http://[host]:[port]/.well-known/agent.json
```

**Example Response:**
```json
{
  "id": "openai-weather-math-agent",
  "name": "OpenAI Weather & Math Agent",
  "description": "An AI agent with weather and calculation capabilities",
  "version": "1.0.0",
  "url": "http://localhost:8000",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  },
  "skills": [
    {
      "id": "get_weather",
      "name": "Get Weather",
      "description": "Get current weather for a location"
    }
  ]
}
```

### 🔧 **Usage Examples**

#### Simple Discovery
```python
import asyncio
from src.client import ObservableA2AClient

async def main():
    # Auto-discover and select first agent
    client = await ObservableA2AClient.discover_and_select(auto_select=True)
    
    if client:
        response = await client.send_message("Hello!")
        print(response)
        await client.close()

asyncio.run(main())
```

#### Advanced Discovery with Filtering
```python
async def find_weather_agents():
    # Find agents with weather capabilities
    weather_filter = {
        "skills": ["get_weather"],
        "capabilities": ["streaming"]
    }
    
    clients = await ObservableA2AClient.discover(
        discovery_scope="local-network",
        agent_filter=weather_filter,
        timeout=10.0
    )
    
    for client in clients:
        print(f"Found weather agent: {client.agent_card.name}")
        # Test weather functionality
        response = await client.send_message("What's the weather in Tokyo?")
        print(response)
        await client.close()
```

#### Interactive Discovery
```python
async def interactive_discovery():
    # Let user choose from discovered agents
    client = await ObservableA2AClient.discover_and_select(
        discovery_scope="localhost"
    )
    
    if client:
        print(f"Connected to: {client.agent_card.name}")
        # Interactive chat session
        while True:
            message = input("You: ")
            if message.lower() == 'quit':
                break
            response = await client.send_message(message)
            print(f"Agent: {response}")
        
        await client.close()
```

### 🚀 **Future Discovery Enhancements**

- **mDNS/Bonjour**: Zero-configuration discovery
- **DNS-SD**: DNS-based service discovery  
- **Registry Services**: Centralized agent marketplaces
- **Blockchain Discovery**: Decentralized agent networks
- **P2P Discovery**: Peer-to-peer agent mesh networks
``` 