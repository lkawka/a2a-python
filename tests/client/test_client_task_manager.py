import pytest
from unittest.mock import AsyncMock, Mock, patch
from a2a.client.client_task_manager import ClientTaskManager
from a2a.client.errors import (
    A2AClientInvalidArgsError,
    A2AClientInvalidStateError,
)
from a2a.types import (
    Task,
    TaskStatus,
    TaskState,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    Message,
    Role,
    Part,
    TextPart,
    Artifact,
)


@pytest.fixture
def task_manager():
    return ClientTaskManager()


@pytest.fixture
def sample_task():
    return Task(
        id='eede470e-ae8f-4910-ba05-085d45dc43c6',
        context_id='9a84655d-3956-4410-ba58-2923639254bd',
        status=TaskStatus(state=TaskState.working),
        history=[],
        artifacts=[],
    )


@pytest.fixture
def sample_message():
    return Message(
        message_id='87c8541d-f773-4825-bbb1-f518727231f2',
        role=Role.user,
        parts=[Part(root=TextPart(text='Hello'))],
    )


def test_get_task_no_task_id_returns_none(task_manager: ClientTaskManager):
    assert task_manager.get_task() is None


def test_get_task_or_raise_no_task_raises_error(
    task_manager: ClientTaskManager,
):
    with pytest.raises(A2AClientInvalidStateError, match='no current Task'):
        task_manager.get_task_or_raise()


@pytest.mark.asyncio
async def test_save_task_event_with_task(
    task_manager: ClientTaskManager, sample_task: Task
):
    await task_manager.save_task_event(sample_task)
    assert task_manager.get_task() == sample_task
    assert task_manager._task_id == sample_task.id
    assert task_manager._context_id == sample_task.context_id


@pytest.mark.asyncio
async def test_save_task_event_with_task_already_set_raises_error(
    task_manager: ClientTaskManager, sample_task: Task
):
    await task_manager.save_task_event(sample_task)
    with pytest.raises(
        A2AClientInvalidArgsError,
        match='Task is already set, create new manager for new tasks.',
    ):
        await task_manager.save_task_event(sample_task)


@pytest.mark.asyncio
async def test_save_task_event_with_status_update(
    task_manager: ClientTaskManager, sample_task: Task, sample_message: Message
):
    await task_manager.save_task_event(sample_task)
    status_update = TaskStatusUpdateEvent(
        task_id=sample_task.id,
        context_id=sample_task.context_id,
        status=TaskStatus(state=TaskState.completed, message=sample_message),
        final=True,
    )
    updated_task = await task_manager.save_task_event(status_update)
    assert updated_task.status.state == TaskState.completed
    assert updated_task.history == [sample_message]


@pytest.mark.asyncio
async def test_save_task_event_with_artifact_update(
    task_manager: ClientTaskManager, sample_task: Task
):
    await task_manager.save_task_event(sample_task)
    artifact = Artifact(
        artifact_id='c9aa91c0-7f03-44dd-8ffe-6d168b48d3d5',
        parts=[Part(root=TextPart(text='artifact content'))],
    )
    artifact_update = TaskArtifactUpdateEvent(
        task_id=sample_task.id,
        context_id=sample_task.context_id,
        artifact=artifact,
    )

    with patch(
        'a2a.client.client_task_manager.append_artifact_to_task'
    ) as mock_append:
        updated_task = await task_manager.save_task_event(artifact_update)
        mock_append.assert_called_once_with(updated_task, artifact_update)


@pytest.mark.asyncio
async def test_save_task_event_creates_task_if_not_exists(
    task_manager: ClientTaskManager,
):
    status_update = TaskStatusUpdateEvent(
        task_id='b3cd0887-5074-4534-8f02-172a36f304a0',
        context_id='7f1ee76a-85ee-406d-856d-b9128ff7bb98',
        status=TaskStatus(state=TaskState.working),
        final=False,
    )
    updated_task = await task_manager.save_task_event(status_update)
    assert updated_task is not None
    assert str(updated_task.id) == 'b3cd0887-5074-4534-8f02-172a36f304a0'
    assert updated_task.status.state == TaskState.working


@pytest.mark.asyncio
async def test_process_with_task_event(
    task_manager: ClientTaskManager, sample_task: Task
):
    with patch.object(
        task_manager, 'save_task_event', new_callable=AsyncMock
    ) as mock_save:
        await task_manager.process(sample_task)
        mock_save.assert_called_once_with(sample_task)


@pytest.mark.asyncio
async def test_process_with_non_task_event(task_manager: ClientTaskManager):
    with patch.object(
        task_manager, 'save_task_event', new_callable=Mock
    ) as mock_save:
        non_task_event = 'not a task event'
        await task_manager.process(non_task_event)
        mock_save.assert_not_called()


def test_update_with_message(
    task_manager: ClientTaskManager, sample_task: Task, sample_message: Message
):
    updated_task = task_manager.update_with_message(sample_message, sample_task)
    assert updated_task.history == [sample_message]


def test_update_with_message_moves_status_message(
    task_manager: ClientTaskManager, sample_task: Task, sample_message: Message
):
    status_message = Message(
        message_id='f82da050-9487-40eb-ae60-a9eda5dc20fc',
        role=Role.agent,
        parts=[Part(root=TextPart(text='Status'))],
    )
    sample_task.status.message = status_message
    updated_task = task_manager.update_with_message(sample_message, sample_task)
    assert updated_task.history == [status_message, sample_message]
    assert updated_task.status.message is None
