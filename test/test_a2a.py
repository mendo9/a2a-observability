#!/usr/bin/env python3
"""
Simple test script to verify A2A implementation works
"""

import asyncio
import os
import sys

# Test imports
try:
    from a2a.types import AgentCard, AgentSkill, AgentCapabilities
    from a2a.server.apps.starlette_app import A2AStarletteApplication
    from a2a.server.agent_execution.agent_executor import AgentExecutor
    from a2a.client import A2AClient
    from a2a.utils.message import new_agent_text_message

    print("✅ All A2A imports successful!")
except ImportError as e:
    print(f"❌ A2A import failed: {e}")
    sys.exit(1)

# Test OpenAI Agents import
try:
    from agents import Agent, Runner, function_tool

    print("✅ OpenAI Agents import successful!")
except ImportError as e:
    print(f"❌ OpenAI Agents import failed: {e}")
    print("Make sure you have OPENAI_API_KEY set and openai-agents installed")

# Test observability imports
try:
    import logfire
    from opentelemetry import trace
    from langfuse import Langfuse

    print("✅ Observability imports successful!")
except ImportError as e:
    print(f"❌ Observability import failed: {e}")


async def test_basic_functionality():
    """Test basic A2A functionality without full server"""

    print("\n🧪 Testing basic A2A functionality...")

    # Test AgentCard creation
    try:
        agent_card = AgentCard(
            id="test-agent",
            name="Test Agent",
            description="A test agent",
            version="1.0.0",
            url="http://localhost:8000",
            capabilities=AgentCapabilities(
                streaming=False,
                pushNotifications=False,
                stateTransitionHistory=False,
            ),
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            skills=[
                AgentSkill(
                    id="test_skill",
                    name="Test Skill",
                    description="A test skill",
                    examples=["test example"],
                    tags=["test", "demo"],
                ),
            ],
        )
        print("✅ AgentCard creation successful!")
    except Exception as e:
        print(f"❌ AgentCard creation failed: {e}")
        return False

    # Test message creation
    try:
        message = new_agent_text_message("Hello, this is a test message!")
        print("✅ Message creation successful!")
    except Exception as e:
        print(f"❌ Message creation failed: {e}")
        return False

    return True


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
    print("\n🎉 Basic tests completed!")
