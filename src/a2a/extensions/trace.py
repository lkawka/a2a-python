from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from a2a._base import A2ABaseModel
from a2a.extensions.base import Extension


class CallTypeEnum(str, Enum):
    """The type of the operation a step represents."""

    AGENT = 'AGENT'
    TOOL = 'TOOL'


class ToolInvocation(A2ABaseModel):
    """A tool invocation."""

    tool_name: str
    parameters: dict[str, Any]


class AgentInvocation(A2ABaseModel):
    """An agent invocation."""

    agent_url: str
    agent_name: str
    requests: dict[str, Any]
    response_trace: ResponseTrace | None = None


class StepAction(A2ABaseModel):
    """The action of a step."""

    tool_invocation: ToolInvocation | None = None
    agent_invocation: AgentInvocation | None = None


class Step(A2ABaseModel):
    """A single operation within a trace."""

    step_id: str
    trace_id: str
    parent_step_id: str | None = None
    call_type: CallTypeEnum
    step_action: StepAction
    cost: int | None = None
    total_tokens: int | None = None
    additional_attributes: dict[str, str] | None = None
    latency: int | None = None
    start_time: datetime
    end_time: datetime


class ResponseTrace(A2ABaseModel):
    """A trace message that contains a collection of spans."""

    trace_id: str
    steps: list[Step]


class TraceExtension(Extension):
    """An extension for traceability."""

    def on_client_message(self, message: Any) -> None:
        """Appends trace information to the message."""
        # This is a placeholder implementation.
        if message.metadata is None:
            message.metadata = {}
        message.metadata['trace'] = 'client-trace'

    def on_server_message(self, message: Any) -> None:
        """Processes trace information from the message."""
        # This is a placeholder implementation.
        if hasattr(message, 'metadata') and 'trace' in message.metadata:
            print(f"Received trace: {message.metadata['trace']}")


AgentInvocation.model_rebuild()