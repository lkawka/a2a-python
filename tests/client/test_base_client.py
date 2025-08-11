from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a.client.base_client import BaseClient
from a2a.client.client import ClientConfig
from a2a.client.transports.base import ClientTransport
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)


@pytest.fixture
def mock_transport():
    transport = AsyncMock(spec=ClientTransport)
    return transport


@pytest.fixture
def sample_agent_card():
    return AgentCard(
        name='Test Agent',
        description='An agent for testing',
        url='http://test.com',
        version='1.0',
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        skills=[],
    )


@pytest.fixture
def sample_message():
    return Message(
        role=Role.user,
        message_id='15957e91-63e6-40ac-8205-1d1ffb09a5b2',
        parts=[Part(root=TextPart(text='Hello'))],
    )


@pytest.fixture
def base_client(sample_agent_card, mock_transport):
    config = ClientConfig(streaming=True)
    return BaseClient(
        card=sample_agent_card,
        config=config,
        transport=mock_transport,
        consumers=[],
        middleware=[],
    )


@pytest.mark.asyncio
async def test_send_message_streaming(
    base_client: BaseClient, mock_transport: MagicMock, sample_message: Message
):
    async def create_stream(*args, **kwargs):
        yield Task(
            id='536ab032-6915-47d1-9909-4172dbee4aa0',
            context_id='9f18b6e9-63c4-4d44-a8b8-f4648003b6b8',
            status=TaskStatus(state=TaskState.completed),
        )

    mock_transport.send_message_streaming.return_value = create_stream()

    events = [event async for event in base_client.send_message(sample_message)]

    mock_transport.send_message_streaming.assert_called_once()
    assert not mock_transport.send_message.called
    assert len(events) == 1
    assert str(events[0][0].id) == '536ab032-6915-47d1-9909-4172dbee4aa0'


@pytest.mark.asyncio
async def test_send_message_non_streaming(
    base_client: BaseClient, mock_transport: MagicMock, sample_message: Message
):
    base_client._config.streaming = False
    mock_transport.send_message.return_value = Task(
        id='9368e3b5-c796-46cf-9318-6c73e1a37e58',
        context_id='0a934875-fa22-4af0-8b40-79b13d46e4a6',
        status=TaskStatus(state=TaskState.completed),
    )

    events = [event async for event in base_client.send_message(sample_message)]

    mock_transport.send_message.assert_called_once()
    assert not mock_transport.send_message_streaming.called
    assert len(events) == 1
    assert str(events[0][0].id) == '9368e3b5-c796-46cf-9318-6c73e1a37e58'


@pytest.mark.asyncio
async def test_send_message_non_streaming_agent_capability_false(
    base_client: BaseClient, mock_transport: MagicMock, sample_message: Message
):
    base_client._card.capabilities.streaming = False
    mock_transport.send_message.return_value = Task(
        id='d7541723-0796-4231-8849-f6f137ea3bf8',
        context_id='dab80cd1-224d-47cd-abd8-cc53101fb273',
        status=TaskStatus(state=TaskState.completed),
    )

    events = [event async for event in base_client.send_message(sample_message)]

    mock_transport.send_message.assert_called_once()
    assert not mock_transport.send_message_streaming.called
    assert len(events) == 1
    assert str(events[0][0].id) == 'd7541723-0796-4231-8849-f6f137ea3bf8'
