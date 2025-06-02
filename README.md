# A2A Observability with Phoenix

A comprehensive A2A (Agent-to-Agent) communication system with full observability using Phoenix, OpenTelemetry, and Docker Compose.

## 🏗️ Architecture

- **Phoenix Server**: Observability platform for AI/LLM applications
- **A2A Server**: Multi-agent system with URL validation and assistant capabilities
- **A2A Client**: Observable client with agent discovery features
- **OpenTelemetry**: Distributed tracing for all components

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- `uv` package manager (for local development)

### 1. Environment Setup

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_BASE_URL=https://api.openai.com/v1
LOGFIRE_TOKEN=your_logfire_token_here
```

### 2. Start with Docker Compose

```bash
# Start Phoenix and A2A Server
docker-compose up -d

# Or start everything including client for testing
docker-compose --profile client up -d
```

### 3. Access the Services

- **Phoenix UI**: http://localhost:6006
- **A2A Server**: http://localhost:8000
- **Agent Card**: http://localhost:8000/.well-known/agent.json

## 🔧 Development Setup

### Local Development (without Docker)

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Start Phoenix server locally**:
   ```bash
   docker run -p 6006:6006 -p 4317:4317 -p 4318:4318 arizephoenix/phoenix:latest
   ```

3. **Run A2A server**:
   ```bash
   uv run python src/server.py
   ```

4. **Run A2A client**:
   ```bash
   uv run python src/client.py
   ```

## 🤖 Multi-Agent System

The A2A server implements a multi-agent system with session management:

### Session States

1. **Validation Phase**: 
   - User must provide a valid URL
   - URL Validation Agent handles validation using `random_testing_tool`
   - Agent looks for URLs starting with `http://` or `https://`

2. **Running Phase**:
   - Main Assistant Agent provides weather and calculation capabilities
   - Access to `get_weather` and `calculate` tools

### Agent Responses

The validation agent uses structured responses:
- **Success**: `VALIDATION_SUCCESS: Your URL has been validated...`
- **Failure**: `VALIDATION_FAILED: Please provide a valid URL...`

### Example Interaction Flow

```
User: https://example.com/test
Agent: VALIDATION_SUCCESS: Your URL has been validated and you can now proceed.
       Great! Now I can help you with weather information, calculations, and other questions.

User: What's the weather in Tokyo?
Agent: The weather in Tokyo is Rainy, 18°C
```

## 📊 Observability Features

### Phoenix Observability

- **Real-time Tracing**: All A2A calls, LLM interactions, and tool usage
- **Session Tracking**: Complete conversation flows with correlation IDs
- **Performance Metrics**: Response times, token usage, error rates

### OpenTelemetry Integration

- **Distributed Tracing**: End-to-end trace correlation
- **Custom Spans**: Agent execution, tool calls, session management
- **Automatic Instrumentation**: HTTP calls, OpenAI SDK calls

### Trace Attributes

Key attributes tracked in traces:
- `session.id`: Session identifier
- `correlation_id`: Request/response correlation
- `agent_type`: url_validation | main_agent
- `tool.name`: Tool being executed
- `input.value`: User input
- `output.value`: Agent response

## 🔍 Agent Discovery

The client supports multiple discovery mechanisms:

### Discovery Methods

1. **Well-Known Endpoints**: `/.well-known/agent.json`
2. **Environment Variables**: `A2A_AGENT_URLS`, `A2A_AGENT_1`, etc.
3. **DNS SRV Records**: `_a2a._tcp.local`
4. **Service Discovery**: Consul, Kubernetes, mDNS
5. **Local Network Scanning**: Automatic port scanning

### Usage Examples

```python
# Discover localhost agents
clients = await ObservableA2AClient.discover(discovery_scope="localhost")

# Auto-select first agent
client = await ObservableA2AClient.discover_and_select(auto_select=True)

# Filter by capabilities
clients = await ObservableA2AClient.discover(
    discovery_scope="localhost",
    agent_filter={"capabilities": ["streaming"]}
)
```

## 🐳 Docker Configuration

### Services

- **phoenix**: Phoenix observability server with persistent data
- **a2a-server**: A2A server with multi-agent system
- **a2a-client**: A2A client for testing (optional profile)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix OTLP endpoint | `http://phoenix:4318/v1/traces` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_BASE_URL` | OpenAI API base URL | `https://api.openai.com/v1` |
| `LOGFIRE_TOKEN` | Logfire token (optional) | - |

### Volumes

- `phoenix_data`: Persistent storage for Phoenix data

## 🔧 Tools Available

### URL Validation Agent Tools

- **random_testing_tool**: Simulates URL validation with random success/failure

### Main Assistant Agent Tools

- **get_weather**: Get weather for any location
  ```
  Example: "What's the weather in New York?"
  Response: "Sunny, 72°F"
  ```

- **calculate**: Basic mathematical operations
  ```
  Example: "Calculate 15 * 7"
  Response: "105"
  ```

## 📈 Monitoring and Debugging

### Phoenix UI Features

- **Traces View**: See all requests and their spans
- **Sessions View**: Track conversation flows
- **Performance View**: Analyze response times and patterns

### Logs and Debugging

- Console logs show trace information
- All HTTP calls are automatically instrumented
- Agent execution details are captured in spans

### Health Checks

- Phoenix: `curl http://localhost:6006/health`
- A2A Server: `curl http://localhost:8000/.well-known/agent.json`

## 🛠️ Troubleshooting

### Common Issues

1. **Phoenix not connecting**:
   - Check if Phoenix container is running: `docker-compose ps`
   - Verify endpoint configuration: `PHOENIX_COLLECTOR_ENDPOINT`

2. **OpenAI API errors**:
   - Verify `OPENAI_API_KEY` is set correctly
   - Check API quota and usage

3. **Agent discovery failing**:
   - Ensure A2A server is accessible
   - Check network connectivity between containers

### Development Tips

- Use `docker-compose logs -f [service]` to see real-time logs
- Phoenix UI shows detailed trace information for debugging
- All traces include correlation IDs for easy debugging

## 📝 Configuration

### Agent Customization

To modify agent behavior, update the instructions in:
- `URLValidationAgentWrapper.__init__()` for validation logic
- `OpenAIAgentWrapper.__init__()` for assistant capabilities

### Adding New Tools

1. Create function with `@function_tool` decorator
2. Add to appropriate agent's tools list
3. Tools are automatically traced with OpenTelemetry

## 🔄 CI/CD and Deployment

The system is designed for easy deployment:

- Docker Compose for local development
- Kubernetes manifests can be generated from Docker Compose
- Environment-based configuration
- Health checks for reliable deployments

## 📚 References

- [Phoenix Documentation](https://arize.com/docs/phoenix)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [A2A SDK Documentation](https://github.com/google/a2a-sdk)
- [OpenAI Agents SDK](https://github.com/openai/agents-sdk) 