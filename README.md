# A2A Observability Demo

A comprehensive implementation of Google's Agent-to-Agent (A2A) protocol using the **official Google A2A SDK** with full observability features including OpenTelemetry, Langfuse, and Logfire.

## 🚀 Features

- **Official Google A2A SDK**: Uses the correct `a2a-sdk` package from Google
- **OpenAI Agents Integration**: Powered by the `openai-agents` SDK
- **Full Observability Stack**:
  - OpenTelemetry for distributed tracing
  - Langfuse for LLM observability
  - Logfire for structured logging
- **Function Tools**: Weather lookup and mathematical calculations
- **Session Management**: Conversation tracking across interactions
- **Error Handling**: Comprehensive error handling and logging

## 📦 Installation

### Prerequisites

- Python 3.13+
- OpenAI API key
- Optional: Langfuse and Logfire accounts for observability

### Setup

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd a2a-observability
   ```

2. **Install dependencies using uv (recommended)**:
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

   Required variables:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Optional observability variables:
   ```bash
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   LOGFIRE_TOKEN=your_logfire_token
   OTEL_EXPORTER_OTLP_ENDPOINT=your_otlp_endpoint
   OTEL_EXPORTER_OTLP_HEADERS=your_otlp_headers
   ```

## 🏃‍♂️ Running the Demo

### Start the A2A Server

```bash
uv run python a2a_obs/a2a_server.py
```

The server will start on `http://localhost:8000` with the following endpoints:
- `http://localhost:8000/.well-known/agent.json` - Agent card endpoint
- `http://localhost:8000/agent/message` - A2A message endpoint

### Run the A2A Client

In a separate terminal:

```bash
uv run python a2a_obs/a2a_client.py
```

Choose between:
1. **Interactive chat** - Chat with the agent in real-time
2. **Automated demo** - Run predefined test scenarios

## 🔧 Available Tools

The A2A agent comes with two built-in tools:

### Weather Tool
```python
get_weather(location: str) -> str
```
Get current weather information for any location.

**Example**: "What's the weather in Tokyo?"

### Calculator Tool
```python
calculate(operation: str, a: float, b: float) -> float
```
Perform basic mathematical operations (add, subtract, multiply, divide).

**Example**: "Calculate 15 * 7"

## 📊 Observability Features

### OpenTelemetry Tracing
- Distributed tracing across A2A calls
- Function tool execution tracing
- HTTP request/response tracing
- Custom span attributes and events

### Langfuse LLM Observability
- LLM call tracking and analysis
- Token usage monitoring
- Conversation flow visualization
- Error tracking and debugging

### Logfire Structured Logging
- Structured log events
- Performance monitoring
- Real-time debugging
- Integration with OpenAI Agents SDK

## 🏗️ Architecture

```
┌─────────────────┐    A2A Protocol    ┌─────────────────┐
│   A2A Client    │◄──────────────────►│   A2A Server    │
│                 │                    │                 │
│ • HTTP Client   │                    │ • OpenAI Agent │
│ • Observability │                    │ • Function Tools│
│ • Session Mgmt  │                    │ • Observability │
└─────────────────┘                    └─────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────┐                    ┌─────────────────┐
│  Observability  │                    │  Observability  │
│                 │                    │                 │
│ • OpenTelemetry │                    │ • OpenTelemetry │
│ • Langfuse      │                    │ • Langfuse      │
│ • HTTP Tracing  │                    │ • Logfire       │
└─────────────────┘                    └─────────────────┘
```

## 🔍 Key Implementation Details

### Correct A2A SDK Usage

The implementation uses the official Google A2A SDK with the agent executor pattern:

```python
# Correct imports
from a2a.types import AgentCard, AgentSkill, Message, TextPart, Role
from a2a.server.apps.starlette_app import A2AStarletteApplication
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.client import A2AClient
```

### Agent Executor Pattern

The A2A SDK uses an agent executor pattern where you implement an `AgentExecutor`:

```python
class ObservableAgentExecutor(AgentExecutor):
    async def execute(self, context: ServerCallContext, event_queue: EventQueue) -> None:
        # Extract message from context
        message = context.params.get("message", {})
        text_content = ""
        for part in message.get("parts", []):
            if part.get("type") == "text":
                text_content += part.get("text", "")
        
        # Process with your agent
        response = await self.agent.invoke(text_content)
        
        # Send response via event queue
        from a2a.server.events.events import new_agent_text_message
        event_queue.enqueue_event(new_agent_text_message(response))
```

### Server Setup

The server uses a Starlette application with proper A2A routing:

```python
# Create agent card
agent_card = AgentCard(
    id="openai-weather-math-agent",
    name="OpenAI Weather & Math Agent",
    description="An AI agent with weather and calculation capabilities",
    skills=[
        AgentSkill(
            id="get_weather",
            name="Get Weather",
            description="Get current weather for a location",
            examples=["What's the weather in Tokyo?"],
        ),
    ],
)

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

# Run with uvicorn
config = uvicorn.Config(app=app.app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()
```

### Client Implementation

The client uses the A2A SDK client with proper message formatting:

```python
client = A2AClient(base_url="http://localhost:8000")

message = Message(
    role=Role.USER,
    parts=[TextPart(type="text", text="Hello, world!")],
)

response = await client.send_message(message=message)
```

## 🧪 Testing

### Interactive Testing

1. Start the server
2. Run the client in interactive mode
3. Try these example queries:
   - "What's the weather in Paris?"
   - "Calculate 25 * 4"
   - "What's the weather in Tokyo and calculate 100 / 5"

### Automated Testing

Run the automated demo to test multiple scenarios:

```bash
uv run python a2a_obs/a2a_client.py
# Choose option 2 for automated demo
```

## 📈 Monitoring and Debugging

### View Traces

1. **OpenTelemetry**: Configure OTLP endpoint to view distributed traces
2. **Langfuse**: Check your Langfuse dashboard for LLM call analysis
3. **Logfire**: Monitor structured logs and performance metrics

### Debug Issues

- Check server logs for A2A protocol errors
- Monitor OpenTelemetry spans for performance bottlenecks
- Use Langfuse to debug LLM interactions
- Review Logfire logs for detailed execution flow

## 🔧 Customization

### Adding New Tools

```python
@function_tool
def my_custom_tool(param: str) -> str:
    """Your custom tool description."""
    # Implementation here
    return result

# Add to agent initialization
self.agent = Agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate, my_custom_tool],
    system_prompt="...",
)
```

### Custom Observability

```python
# Add custom spans
with tracer.start_as_current_span("custom_operation") as span:
    span.set_attribute("custom_attribute", value)
    # Your code here
```

## 🚨 Common Issues

### Import Errors
- Ensure you're using `a2a-sdk` not `a2a` or `python-a2a`
- Check that all dependencies are installed correctly

### Connection Issues
- Verify the server is running on the correct port
- Check firewall settings
- Ensure environment variables are set correctly

### Observability Issues
- Verify API keys are correct
- Check network connectivity to observability platforms
- Review environment variable configuration

## 📚 References

- [Google A2A Protocol](https://github.com/google-a2a/a2a-python)
- [Official A2A SDK](https://pypi.org/project/a2a-sdk/)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Logfire Documentation](https://logfire.pydantic.dev/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This implementation uses the official Google A2A SDK and follows the correct protocol specifications. Previous implementations using incorrect imports have been fixed to ensure compatibility with the official A2A ecosystem. 