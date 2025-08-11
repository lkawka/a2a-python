import asyncio
import logging
import time

from unittest.mock import (
    AsyncMock,
    MagicMock,
    PropertyMock,
    patch,
)

import pytest

from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
    RequestContextBuilder,
    SimpleRequestContextBuilder,
)
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue, InMemoryQueueManager, QueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    ResultAggregator,
    TaskStore,
    TaskUpdater,
)
from a2a.types import (
    DeleteTaskPushNotificationConfigParams,
    GetTaskPushNotificationConfigParams,
    InternalError,
    InvalidParamsError,
    ListTaskPushNotificationConfigParams,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    Role,
    Task,
    TaskIdParams,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_task,
)


class DummyAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task_updater = TaskUpdater(
            event_queue, context.task_id, context.context_id
        )
        async for i in self._run():
            parts = [Part(root=TextPart(text=f'Event {i}'))]
            try:
                await task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(parts),
                )
            except RuntimeError:
                # Stop processing when the event loop is closed
                break

    async def _run(self):
        for i in range(1_000_000):  # Simulate a long-running stream
            yield i

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


# Helper to create a simple task for tests
def create_sample_task(
    task_id='13d5b8a8-62d7-4490-98c8-d3951b42702a',
    status_state=TaskState.submitted,
    context_id='06cc947f-8946-4bde-b776-165462407e57',
) -> Task:
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=status_state),
    )


# Helper to create ServerCallContext
def create_server_call_context() -> ServerCallContext:
    # Assuming UnauthenticatedUser is available or can be imported
    from a2a.auth.user import UnauthenticatedUser

    return ServerCallContext(user=UnauthenticatedUser())


def test_init_default_dependencies():
    """Test that default dependencies are created if not provided."""
    agent_executor = DummyAgentExecutor()
    task_store = InMemoryTaskStore()

    handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=task_store
    )

    assert isinstance(handler._queue_manager, InMemoryQueueManager)
    assert isinstance(
        handler._request_context_builder, SimpleRequestContextBuilder
    )
    assert handler._push_config_store is None
    assert handler._push_sender is None
    assert (
        handler._request_context_builder._should_populate_referred_tasks
        is False
    )
    assert handler._request_context_builder._task_store == task_store


@pytest.mark.asyncio
async def test_on_get_task_not_found():
    """Test on_get_task when task_store.get returns None."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = None

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(), task_store=mock_task_store
    )

    params = TaskQueryParams(id='b87b95a6-cf02-4d0a-8355-eb9cf307d323')

    from a2a.utils.errors import ServerError  # Local import for ServerError

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_get_task(params, create_server_call_context())

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    mock_task_store.get.assert_awaited_once_with(
        'b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )


@pytest.mark.asyncio
async def test_on_cancel_task_task_not_found():
    """Test on_cancel_task when the task is not found."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = None

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(), task_store=mock_task_store
    )
    params = TaskIdParams(id='d1acbc3d-f320-4c99-aee9-3c9c9627c3f6')

    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_cancel_task(
            params, create_server_call_context()
        )

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    mock_task_store.get.assert_awaited_once_with(
        'd1acbc3d-f320-4c99-aee9-3c9c9627c3f6'
    )


@pytest.mark.asyncio
async def test_on_cancel_task_queue_tap_returns_none():
    """Test on_cancel_task when queue_manager.tap returns None."""
    mock_task_store = AsyncMock(spec=TaskStore)
    sample_task = create_sample_task(
        task_id='82680a9b-d9cf-40f6-bf17-ff03f32accb5'
    )
    mock_task_store.get.return_value = sample_task

    mock_queue_manager = AsyncMock(spec=QueueManager)
    mock_queue_manager.tap.return_value = (
        None  # Simulate queue not found / tap returns None
    )

    mock_agent_executor = AsyncMock(
        spec=AgentExecutor
    )  # Use AsyncMock for agent_executor

    # Mock ResultAggregator and its consume_all method
    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    mock_result_aggregator_instance.consume_all.return_value = (
        create_sample_task(
            task_id='82680a9b-d9cf-40f6-bf17-ff03f32accb5',
            status_state=TaskState.canceled,  # Expected final state
        )
    )

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        queue_manager=mock_queue_manager,
    )

    with patch(
        'a2a.server.request_handlers.default_request_handler.ResultAggregator',
        return_value=mock_result_aggregator_instance,
    ):
        params = TaskIdParams(id='82680a9b-d9cf-40f6-bf17-ff03f32accb5')
        result_task = await request_handler.on_cancel_task(
            params, create_server_call_context()
        )

    mock_task_store.get.assert_awaited_once_with(
        '82680a9b-d9cf-40f6-bf17-ff03f32accb5'
    )
    mock_queue_manager.tap.assert_awaited_once_with(
        '82680a9b-d9cf-40f6-bf17-ff03f32accb5'
    )
    # agent_executor.cancel should be called with a new EventQueue if tap returned None
    mock_agent_executor.cancel.assert_awaited_once()
    # Verify the EventQueue passed to cancel was a new one
    call_args_list = mock_agent_executor.cancel.call_args_list
    args, _ = call_args_list[0]
    assert isinstance(
        args[1], EventQueue
    )  # args[1] is the event_queue argument

    mock_result_aggregator_instance.consume_all.assert_awaited_once()
    assert result_task is not None
    assert result_task.status.state == TaskState.canceled


@pytest.mark.asyncio
async def test_on_cancel_task_cancels_running_agent():
    """Test on_cancel_task cancels a running agent task."""
    task_id = 'e5681ebe-445c-49e1-b956-ee7c155c92f2'
    sample_task = create_sample_task(task_id=task_id)
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = sample_task

    mock_queue_manager = AsyncMock(spec=QueueManager)
    mock_event_queue = AsyncMock(spec=EventQueue)
    mock_queue_manager.tap.return_value = mock_event_queue

    mock_agent_executor = AsyncMock(spec=AgentExecutor)

    # Mock ResultAggregator
    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    mock_result_aggregator_instance.consume_all.return_value = (
        create_sample_task(task_id=task_id, status_state=TaskState.canceled)
    )

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        queue_manager=mock_queue_manager,
    )

    # Simulate a running agent task
    mock_producer_task = AsyncMock(spec=asyncio.Task)
    request_handler._running_agents[task_id] = mock_producer_task

    with patch(
        'a2a.server.request_handlers.default_request_handler.ResultAggregator',
        return_value=mock_result_aggregator_instance,
    ):
        params = TaskIdParams(id=task_id)
        await request_handler.on_cancel_task(
            params, create_server_call_context()
        )

    mock_producer_task.cancel.assert_called_once()
    mock_agent_executor.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_cancel_task_invalid_result_type():
    """Test on_cancel_task when result_aggregator returns a Message instead of a Task."""
    task_id = 'e2b71d7b-f826-4768-9b02-f1ec851aac19'
    sample_task = create_sample_task(task_id=task_id)
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = sample_task

    mock_queue_manager = AsyncMock(spec=QueueManager)
    mock_event_queue = AsyncMock(spec=EventQueue)
    mock_queue_manager.tap.return_value = mock_event_queue

    mock_agent_executor = AsyncMock(spec=AgentExecutor)

    # Mock ResultAggregator to return a Message
    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    mock_result_aggregator_instance.consume_all.return_value = Message(
        message_id='6e7579fb-eeeb-439d-8b94-0008c6a2c626',
        role=Role.agent,
        parts=[],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        queue_manager=mock_queue_manager,
    )

    from a2a.utils.errors import ServerError  # Local import

    with patch(
        'a2a.server.request_handlers.default_request_handler.ResultAggregator',
        return_value=mock_result_aggregator_instance,
    ):
        params = TaskIdParams(id=task_id)
        with pytest.raises(ServerError) as exc_info:
            await request_handler.on_cancel_task(
                params, create_server_call_context()
            )

    assert isinstance(exc_info.value.error, InternalError)
    assert (
        'Agent did not return valid response for cancel'
        in exc_info.value.error.message
    )  # type: ignore


@pytest.mark.asyncio
async def test_on_message_send_with_push_notification():
    """Test on_message_send sets push notification info if provided."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_push_notification_store = AsyncMock(spec=PushNotificationConfigStore)
    mock_agent_executor = AsyncMock(spec=AgentExecutor)
    mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

    task_id = '757b7ee2-0cfa-4234-b436-d65fc65acb22'
    context_id = 'f0b24c69-a41b-4a56-8c5a-0f8d54b6c09b'
    sample_initial_task = create_sample_task(
        task_id=task_id, context_id=context_id, status_state=TaskState.submitted
    )

    # TaskManager will be created inside on_message_send.
    # We need to mock task_store.get to return None initially for TaskManager to create a new task.
    # Then, TaskManager.update_with_message will be called.
    # For simplicity in this unit test, let's assume TaskManager correctly sets up the task
    # and the task object (with IDs) is available for _request_context_builder.build

    mock_task_store.get.return_value = (
        None  # Simulate new task scenario for TaskManager
    )

    # Mock _request_context_builder.build to return a context with the generated/confirmed IDs
    mock_request_context = MagicMock(spec=RequestContext)
    mock_request_context.task_id = task_id
    mock_request_context.context_id = context_id
    mock_request_context_builder.build.return_value = mock_request_context

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        push_config_store=mock_push_notification_store,
        request_context_builder=mock_request_context_builder,
    )

    push_config = PushNotificationConfig(url='http://callback.com/push')
    message_config = MessageSendConfiguration(
        push_notification_config=push_config,
        accepted_output_modes=['text/plain'],  # Added required field
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='5b7d1d17-173b-4b55-ab83-3e5b566fc104',
            parts=[],
            task_id=task_id,
            context_id=context_id,
        ),
        configuration=message_config,
    )

    # Mock ResultAggregator and its consume_and_break_on_interrupt
    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    final_task_result = create_sample_task(
        task_id=task_id, context_id=context_id, status_state=TaskState.completed
    )
    mock_result_aggregator_instance.consume_and_break_on_interrupt.return_value = (
        final_task_result,
        False,
    )

    # Mock the current_result property to return the final task result
    async def get_current_result():
        return final_task_result

    # Configure the 'current_result' property on the type of the mock instance
    type(mock_result_aggregator_instance).current_result = PropertyMock(
        return_value=get_current_result()
    )

    with (
        patch(
            'a2a.server.request_handlers.default_request_handler.ResultAggregator',
            return_value=mock_result_aggregator_instance,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
            return_value=sample_initial_task,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.update_with_message',
            return_value=sample_initial_task,
        ),
    ):  # Ensure task object is returned
        await request_handler.on_message_send(
            params, create_server_call_context()
        )

    mock_push_notification_store.set_info.assert_awaited_once_with(
        task_id, push_config
    )
    # Other assertions for full flow if needed (e.g., agent execution)
    mock_agent_executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_message_send_with_push_notification_no_existing_Task():
    """Test on_message_send for new task sets push notification info if provided."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_push_notification_store = AsyncMock(spec=PushNotificationConfigStore)
    mock_agent_executor = AsyncMock(spec=AgentExecutor)
    mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

    task_id = '757b7ee2-0cfa-4234-b436-d65fc65acb22'
    context_id = 'f0b24c69-a41b-4a56-8c5a-0f8d54b6c09b'

    mock_task_store.get.return_value = (
        None  # Simulate new task scenario for TaskManager
    )

    # Mock _request_context_builder.build to return a context with the generated/confirmed IDs
    mock_request_context = MagicMock(spec=RequestContext)
    mock_request_context.task_id = task_id
    mock_request_context.context_id = context_id
    mock_request_context_builder.build.return_value = mock_request_context

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        push_config_store=mock_push_notification_store,
        request_context_builder=mock_request_context_builder,
    )

    push_config = PushNotificationConfig(url='http://callback.com/push')
    message_config = MessageSendConfiguration(
        push_notification_config=push_config,
        accepted_output_modes=['text/plain'],  # Added required field
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='5b7d1d17-173b-4b55-ab83-3e5b566fc104',
            parts=[],
        ),
        configuration=message_config,
    )

    # Mock ResultAggregator and its consume_and_break_on_interrupt
    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    final_task_result = create_sample_task(
        task_id=task_id, context_id=context_id, status_state=TaskState.completed
    )
    mock_result_aggregator_instance.consume_and_break_on_interrupt.return_value = (
        final_task_result,
        False,
    )

    # Mock the current_result property to return the final task result
    async def get_current_result():
        return final_task_result

    # Configure the 'current_result' property on the type of the mock instance
    type(mock_result_aggregator_instance).current_result = PropertyMock(
        return_value=get_current_result()
    )

    with (
        patch(
            'a2a.server.request_handlers.default_request_handler.ResultAggregator',
            return_value=mock_result_aggregator_instance,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
            return_value=None,
        ),
    ):
        await request_handler.on_message_send(
            params, create_server_call_context()
        )

    mock_push_notification_store.set_info.assert_awaited_once_with(
        task_id, push_config
    )
    # Other assertions for full flow if needed (e.g., agent execution)
    mock_agent_executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_message_send_no_result_from_aggregator():
    """Test on_message_send when aggregator returns (None, False)."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_agent_executor = AsyncMock(spec=AgentExecutor)
    mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

    task_id = '43b91701-0e3e-415b-b4bf-2d53eb23db81'
    # Mock _request_context_builder.build
    mock_request_context = MagicMock(spec=RequestContext)
    mock_request_context.task_id = task_id
    mock_request_context_builder.build.return_value = mock_request_context

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        request_context_builder=mock_request_context_builder,
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='0df4ec66-b5d7-4872-ba3a-5f7b231a9022',
            parts=[],
        )
    )

    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    mock_result_aggregator_instance.consume_and_break_on_interrupt.return_value = (
        None,
        False,
    )

    from a2a.utils.errors import ServerError  # Local import

    with (
        patch(
            'a2a.server.request_handlers.default_request_handler.ResultAggregator',
            return_value=mock_result_aggregator_instance,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
            return_value=None,
        ),
    ):  # TaskManager.get_task for initial task
        with pytest.raises(ServerError) as exc_info:
            await request_handler.on_message_send(
                params, create_server_call_context()
            )

    assert isinstance(exc_info.value.error, InternalError)


@pytest.mark.asyncio
async def test_on_message_send_task_id_mismatch():
    """Test on_message_send when result task ID doesn't match request context task ID."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_agent_executor = AsyncMock(spec=AgentExecutor)
    mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

    context_task_id = 'c65e1186-487b-4bd2-a7f3-ddc766a02783'
    result_task_id = 'b6d8d5df-2da3-4f86-a173-388ed8654584'  # Mismatch

    # Mock _request_context_builder.build
    mock_request_context = MagicMock(spec=RequestContext)
    mock_request_context.task_id = context_task_id
    mock_request_context_builder.build.return_value = mock_request_context

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        request_context_builder=mock_request_context_builder,
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='2cf42a1c-b61c-44ed-88f1-58111760e21f',
            parts=[],
        )
    )

    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    mismatched_task = create_sample_task(task_id=result_task_id)
    mock_result_aggregator_instance.consume_and_break_on_interrupt.return_value = (
        mismatched_task,
        False,
    )

    from a2a.utils.errors import ServerError  # Local import

    with (
        patch(
            'a2a.server.request_handlers.default_request_handler.ResultAggregator',
            return_value=mock_result_aggregator_instance,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
            return_value=None,
        ),
    ):
        with pytest.raises(ServerError) as exc_info:
            await request_handler.on_message_send(
                params, create_server_call_context()
            )

    assert isinstance(exc_info.value.error, InternalError)
    assert 'Task ID mismatch' in exc_info.value.error.message  # type: ignore


class HelloAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task = context.current_task
        if not task:
            assert context.message is not None, (
                'A message is required to create a new task'
            )
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            parts = [Part(root=TextPart(text='I am working'))]
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message(parts),
            )
        except Exception as e:
            # Stop processing when the event loop is closed
            logging.warning('Error: %s', e)
            return
        await updater.add_artifact(
            [Part(root=TextPart(text='Hello world!'))],
            name='conversion_result',
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


@pytest.mark.asyncio
async def test_on_message_send_non_blocking():
    task_store = InMemoryTaskStore()
    push_store = InMemoryPushNotificationConfigStore()

    request_handler = DefaultRequestHandler(
        agent_executor=HelloAgentExecutor(),
        task_store=task_store,
        push_config_store=push_store,
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='5b7d1d17-173b-4b55-ab83-3e5b566fc104',
            parts=[Part(root=TextPart(text='Hi'))],
        ),
        configuration=MessageSendConfiguration(
            blocking=False, accepted_output_modes=['text/plain']
        ),
    )

    result = await request_handler.on_message_send(
        params, create_server_call_context()
    )

    assert result is not None
    assert isinstance(result, Task)
    assert result.status.state == TaskState.submitted

    # Polling for 500ms until task is completed.
    task: Task | None = None
    for _ in range(5):
        await asyncio.sleep(0.1)
        task = await task_store.get(result.id)
        assert task is not None
        if task.status.state == TaskState.completed:
            break

    assert task is not None
    assert task.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_on_message_send_interrupted_flow():
    """Test on_message_send when flow is interrupted (e.g., auth_required)."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_agent_executor = AsyncMock(spec=AgentExecutor)
    mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

    task_id = '6a24e838-b6cb-4ccb-aab1-09b57ec28256'
    # Mock _request_context_builder.build
    mock_request_context = MagicMock(spec=RequestContext)
    mock_request_context.task_id = task_id
    mock_request_context_builder.build.return_value = mock_request_context

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        request_context_builder=mock_request_context_builder,
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='21b34182-195d-467a-9ed5-3160c8acd8f5',
            parts=[],
        )
    )

    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    interrupt_task_result = create_sample_task(
        task_id=task_id, status_state=TaskState.auth_required
    )
    mock_result_aggregator_instance.consume_and_break_on_interrupt.return_value = (
        interrupt_task_result,
        True,
    )  # Interrupted = True

    # Patch asyncio.create_task to verify _cleanup_producer is scheduled
    with (
        patch('asyncio.create_task') as mock_asyncio_create_task,
        patch(
            'a2a.server.request_handlers.default_request_handler.ResultAggregator',
            return_value=mock_result_aggregator_instance,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
            return_value=None,
        ),
    ):
        result = await request_handler.on_message_send(
            params, create_server_call_context()
        )

    assert result == interrupt_task_result
    assert (
        mock_asyncio_create_task.call_count == 2
    )  # First for _run_event_stream, second for _cleanup_producer

    # Check that the second call to create_task was for _cleanup_producer
    found_cleanup_call = False
    for call_args_tuple in mock_asyncio_create_task.call_args_list:
        created_coro = call_args_tuple[0][0]
        if (
            hasattr(created_coro, '__name__')
            and created_coro.__name__ == '_cleanup_producer'
        ):
            found_cleanup_call = True
            break
    assert found_cleanup_call, (
        '_cleanup_producer was not scheduled with asyncio.create_task'
    )


@pytest.mark.asyncio
async def test_on_message_send_stream_with_push_notification():
    """Test on_message_send_stream sets and uses push notification info."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_push_config_store = AsyncMock(spec=PushNotificationConfigStore)
    mock_push_sender = AsyncMock(spec=PushNotificationSender)
    mock_agent_executor = AsyncMock(spec=AgentExecutor)
    mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

    task_id = '13662c2a-8544-4fd0-b052-60797780e15a'
    context_id = '3587eea8-d5e6-41c1-9aa9-9d8acb24797f'

    # Initial task state for TaskManager
    initial_task_for_tm = create_sample_task(
        task_id=task_id, context_id=context_id, status_state=TaskState.submitted
    )

    # Task state for RequestContext
    task_for_rc = create_sample_task(
        task_id=task_id, context_id=context_id, status_state=TaskState.working
    )  # Example state after message update

    mock_task_store.get.return_value = None  # New task for TaskManager

    mock_request_context = MagicMock(spec=RequestContext)
    mock_request_context.task_id = task_id
    mock_request_context.context_id = context_id
    mock_request_context_builder.build.return_value = mock_request_context

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        push_config_store=mock_push_config_store,
        push_sender=mock_push_sender,
        request_context_builder=mock_request_context_builder,
    )

    push_config = PushNotificationConfig(url='http://callback.stream.com/push')
    message_config = MessageSendConfiguration(
        push_notification_config=push_config,
        accepted_output_modes=['text/plain'],  # Added required field
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='cffc8e42-0016-40e1-99ba-3e588923312c',
            parts=[],
            task_id=task_id,
            context_id=context_id,
        ),
        configuration=message_config,
    )

    # Mock ResultAggregator and its consume_and_emit
    mock_result_aggregator_instance = MagicMock(
        spec=ResultAggregator
    )  # Use MagicMock for easier property mocking

    # Events to be yielded by consume_and_emit
    event1_task_update = create_sample_task(
        task_id=task_id, context_id=context_id, status_state=TaskState.working
    )
    event2_final_task = create_sample_task(
        task_id=task_id, context_id=context_id, status_state=TaskState.completed
    )

    async def event_stream_gen():
        yield event1_task_update
        yield event2_final_task

    # consume_and_emit is called by `async for ... in result_aggregator.consume_and_emit(consumer)`
    # This means result_aggregator.consume_and_emit(consumer) must directly return an async iterable.
    # If consume_and_emit is an async method, this is problematic in the product code.
    # For the test, we make the mock of consume_and_emit a synchronous method
    # that returns the async generator object.
    def sync_get_event_stream_gen(*args, **kwargs):
        return event_stream_gen()

    mock_result_aggregator_instance.consume_and_emit = MagicMock(
        side_effect=sync_get_event_stream_gen
    )

    # Mock current_result property to return appropriate awaitables
    # Coroutines that will be returned by successive accesses to current_result
    async def current_result_coro1():
        return event1_task_update

    async def current_result_coro2():
        return event2_final_task

    # Use unittest.mock.PropertyMock for async property
    # We need to patch 'ResultAggregator.current_result' when this instance is used.
    # This is complex because ResultAggregator is instantiated inside the handler.
    # Easier: If mock_result_aggregator_instance is a MagicMock, we can assign a callable.
    # This part is tricky. Let's assume current_result is an async method for easier mocking first.
    # If it's truly a property, the mocking is harder with instance mocks.
    # Let's adjust the mock_result_aggregator_instance.current_result to be an AsyncMock directly
    # This means the code would call `await result_aggregator.current_result()`
    # But the actual code is `await result_aggregator.current_result`
    # This implies `result_aggregator.current_result` IS an awaitable.
    # So, we can mock it with a side_effect that returns awaitables (coroutines).

    # Create simple awaitables (coroutines) for side_effect
    async def get_event1():
        return event1_task_update

    async def get_event2():
        return event2_final_task

    # Make the current_result attribute of the mock instance itself an awaitable
    # This still means current_result is not callable.
    # For an async property, the mock needs to have current_result as a non-AsyncMock attribute
    # that is itself an awaitable.

    # Let's try to mock the property at the type level for ResultAggregator temporarily
    # This is not ideal as it affects all instances.

    # Alternative: Configure the AsyncMock for current_result to return a coroutine
    # when it's awaited. This is not directly supported by AsyncMock for property access.

    # Simplest for now: Assume `current_result` attribute of the mocked `ResultAggregator` instance
    # can be sequentially awaited if it's a list of awaitables that a test runner can handle.
    # This is likely to fail again but will clarify the exact point of await.
    # The error "TypeError: object AsyncMock can't be used in 'await' expression" means
    # `mock_result_aggregator_instance.current_result` is an AsyncMock, and that's what's awaited.
    # This AsyncMock needs to have a __await__ method.

    # Let's make the side_effect of the AsyncMock `current_result` provide the values.
    # This assumes that `await mock.property` somehow triggers a call to the mock.
    # This is not how AsyncMock works.

    # The code is `await result_aggregator.current_result`.
    # `result_aggregator` is an instance of `ResultAggregator`.
    # `current_result` is an async property.
    # So `result_aggregator.current_result` evaluates to a coroutine.
    # We need `mock_result_aggregator_instance.current_result` to be a coroutine,
    # or a list of coroutines if accessed multiple times.
    # This is best done by mocking the property itself.
    # Let's assume it's called twice.

    # We will patch ResultAggregator to be our mock_result_aggregator_instance
    # Then, we need to control what its `current_result` property returns.
    # We can use a PropertyMock for this, attached to the type of mock_result_aggregator_instance.

    # For this specific test, let's make current_result a simple async def method on the mock instance
    # This means we are slightly diverging from the "property" nature just for this mock.
    # Mock current_result property to return appropriate awaitables (coroutines) sequentially.
    async def get_event1_coro():
        return event1_task_update

    async def get_event2_coro():
        return event2_final_task

    # Configure the 'current_result' property on the type of the mock instance
    # This makes accessing `instance.current_result` call the side_effect function,
    # which then cycles through our list of coroutines.
    # We need a new PropertyMock for each instance, or patch the class.
    # Since mock_result_aggregator_instance is already created, we attach to its type.
    # This can be tricky. A more direct way is to ensure the instance's attribute `current_result`
    # behaves as desired. If `mock_result_aggregator_instance` is a `MagicMock`, its attributes are also mocks.

    # Let's make `current_result` a MagicMock whose side_effect returns the coroutines.
    # This means when `result_aggregator.current_result` is accessed, this mock is "called".
    # This isn't quite right for a property. A property isn't "called" on access.

    # Correct approach for mocking an async property on an instance mock:
    # Set the attribute `current_result` on the instance `mock_result_aggregator_instance`
    # to be a `PropertyMock` if we were patching the class.
    # Since we have the instance, we can try to replace its `current_result` attribute.
    # The instance `mock_result_aggregator_instance` is a `MagicMock`.
    # We can make `mock_result_aggregator_instance.current_result` a `PropertyMock`
    # that returns a coroutine. For multiple calls, `side_effect` on `PropertyMock` is a list of return_values.

    # Create a PropertyMock that will cycle through coroutines
    # This requires Python 3.8+ for PropertyMock to be directly usable with side_effect list for properties.
    # For older versions or for clarity with async properties, directly mocking the attribute
    # to be a series of awaitables is hard.
    # The easiest is to ensure `current_result` is an AsyncMock that returns the values.
    # The product code `await result_aggregator.current_result` means `current_result` must be an awaitable.

    # Let's make current_result an AsyncMock whose __call__ returns the sequence.
    # Mock current_result as an async property
    # Create coroutines that will be the "result" of awaiting the property
    async def get_current_result_coro1():
        return event1_task_update

    async def get_current_result_coro2():
        return event2_final_task

    # Configure the 'current_result' property on the mock_result_aggregator_instance
    # using PropertyMock attached to its type. This makes instance.current_result return
    # items from side_effect sequentially on each access.
    # Since current_result is an async property, these items should be coroutines.
    # We need to ensure that mock_result_aggregator_instance itself is the one patched.
    # The patch for ResultAggregator returns this instance.
    # So, we configure PropertyMock on the type of this specific mock instance.
    # This is slightly unusual; typically PropertyMock is used when patching a class.
    # A more straightforward approach for an instance is if its type is already a mock.
    # As mock_result_aggregator_instance is a MagicMock, we can configure its 'current_result'
    # attribute to be a PropertyMock.

    # Let's directly assign a PropertyMock to the type of the instance for `current_result`
    # This ensures that when `instance.current_result` is accessed, the PropertyMock's logic is triggered.
    # However, PropertyMock is usually used with `patch.object` or by setting it on the class.
    #
    # A simpler way for MagicMock instance:
    # `mock_result_aggregator_instance.current_result` is already a MagicMock (or AsyncMock if spec'd).
    # We need to make it return a coroutine upon access.
    # The most direct way to mock an async property on a MagicMock instance
    # such that it returns a sequence of awaitables:
    async def side_effect_current_result():
        yield event1_task_update
        yield event2_final_task

    # Create an async generator from the side effect
    current_result_gen = side_effect_current_result()

    # Make current_result return the next item from this generator (wrapped in a coroutine)
    # each time it's accessed.
    async def get_next_current_result():
        try:
            return await current_result_gen.__anext__()
        except StopAsyncIteration:
            # Handle case where it's awaited more times than values provided
            return None  # Or raise an error

    # Since current_result is a property, accessing it should return a coroutine.
    # We can achieve this by making mock_result_aggregator_instance.current_result
    # a MagicMock whose side_effect returns these coroutines.
    # This is still tricky because it's a property access.

    # Let's use the PropertyMock on the class being mocked via the patch.
    # Setup for consume_and_emit
    def sync_get_event_stream_gen_for_prop_test(*args, **kwargs):
        return event_stream_gen()

    mock_result_aggregator_instance.consume_and_emit = MagicMock(
        side_effect=sync_get_event_stream_gen_for_prop_test
    )

    # Configure current_result on the type of the mock_result_aggregator_instance
    # This makes it behave like a property that returns items from side_effect on access.
    type(mock_result_aggregator_instance).current_result = PropertyMock(
        side_effect=[get_current_result_coro1(), get_current_result_coro2()]
    )

    with (
        patch(
            'a2a.server.request_handlers.default_request_handler.ResultAggregator',
            return_value=mock_result_aggregator_instance,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
            return_value=initial_task_for_tm,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.update_with_message',
            return_value=task_for_rc,
        ),
    ):
        # Consume the stream
        async for _ in request_handler.on_message_send_stream(
            params, create_server_call_context()
        ):
            pass

    # Assertions
    # 1. set_info called once at the beginning if task exists (or after task is created from message)
    mock_push_config_store.set_info.assert_any_call(task_id, push_config)

    # 2. send_notification called for each task event yielded by aggregator
    assert mock_push_sender.send_notification.await_count == 2
    mock_push_sender.send_notification.assert_any_await(event1_task_update)
    mock_push_sender.send_notification.assert_any_await(event2_final_task)

    mock_agent_executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_message_send_stream_task_id_mismatch():
    """Test on_message_send_stream raises error if yielded task ID mismatches."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_agent_executor = AsyncMock(
        spec=AgentExecutor
    )  # Only need a basic mock
    mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

    context_task_id = 'dc1d1c1e-0b5b-45d1-a70e-47faafb2adff'
    mismatched_task_id = '677cb217-5841-475c-9f45-06c3f8914f65'

    mock_request_context = MagicMock(spec=RequestContext)
    mock_request_context.task_id = context_task_id
    mock_request_context_builder.build.return_value = mock_request_context

    request_handler = DefaultRequestHandler(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        request_context_builder=mock_request_context_builder,
    )
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='b990eddf-01d6-414d-90cb-02f8549debe9',
            parts=[],
        )
    )

    mock_result_aggregator_instance = AsyncMock(spec=ResultAggregator)
    mismatched_task_event = create_sample_task(
        task_id=mismatched_task_id
    )  # Task with different ID

    async def event_stream_gen_mismatch():
        yield mismatched_task_event

    mock_result_aggregator_instance.consume_and_emit.return_value = (
        event_stream_gen_mismatch()
    )

    from a2a.utils.errors import ServerError  # Local import

    with (
        patch(
            'a2a.server.request_handlers.default_request_handler.ResultAggregator',
            return_value=mock_result_aggregator_instance,
        ),
        patch(
            'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
            return_value=None,
        ),
    ):
        with pytest.raises(ServerError) as exc_info:
            async for _ in request_handler.on_message_send_stream(
                params, create_server_call_context()
            ):
                pass  # Consume the stream to trigger the error

    assert isinstance(exc_info.value.error, InternalError)
    assert 'Task ID mismatch' in exc_info.value.error.message  # type: ignore


@pytest.mark.asyncio
async def test_cleanup_producer_task_id_not_in_running_agents():
    """Test _cleanup_producer when task_id is not in _running_agents (e.g., already cleaned up)."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_queue_manager = AsyncMock(spec=QueueManager)
    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        queue_manager=mock_queue_manager,
    )

    task_id = 'f7075439-b5c9-4286-a09a-a7a2d2756a03'

    # Create a real, completed asyncio.Task for the test
    async def dummy_coro_for_task():
        pass

    mock_producer_task = asyncio.create_task(dummy_coro_for_task())
    await asyncio.sleep(
        0
    )  # Ensure the task has a chance to complete/be scheduled

    # Call cleanup directly, ensuring task_id is NOT in _running_agents
    # This simulates a race condition or double cleanup.
    if task_id in request_handler._running_agents:
        del request_handler._running_agents[task_id]  # Ensure it's not there

    try:
        await request_handler._cleanup_producer(mock_producer_task, task_id)
    except Exception as e:
        pytest.fail(f'_cleanup_producer raised an exception unexpectedly: {e}')

    # Verify queue_manager.close was still called
    mock_queue_manager.close.assert_awaited_once_with(task_id)
    # No error should be raised by pop if key is missing and default is None.


@pytest.mark.asyncio
async def test_set_task_push_notification_config_no_notifier():
    """Test on_set_task_push_notification_config when _push_config_store is None."""
    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=AsyncMock(spec=TaskStore),
        push_config_store=None,  # Explicitly None
    )
    params = TaskPushNotificationConfig(
        task_id='13d5b8a8-62d7-4490-98c8-d3951b42702a',
        push_notification_config=PushNotificationConfig(
            url='http://example.com'
        ),
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_set_task_push_notification_config(
            params, create_server_call_context()
        )
    assert isinstance(exc_info.value.error, UnsupportedOperationError)


@pytest.mark.asyncio
async def test_set_task_push_notification_config_task_not_found():
    """Test on_set_task_push_notification_config when task is not found."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = None  # Task not found
    mock_push_store = AsyncMock(spec=PushNotificationConfigStore)
    mock_push_sender = AsyncMock(spec=PushNotificationSender)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=mock_push_store,
        push_sender=mock_push_sender,
    )
    params = TaskPushNotificationConfig(
        task_id='b87b95a6-cf02-4d0a-8355-eb9cf307d323',
        push_notification_config=PushNotificationConfig(
            url='http://example.com'
        ),
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_set_task_push_notification_config(
            params, create_server_call_context()
        )

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    mock_task_store.get.assert_awaited_once_with(
        'b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_push_store.set_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_task_push_notification_config_no_store():
    """Test on_get_task_push_notification_config when _push_config_store is None."""
    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=AsyncMock(spec=TaskStore),
        push_config_store=None,  # Explicitly None
    )
    params = GetTaskPushNotificationConfigParams(
        id='13d5b8a8-62d7-4490-98c8-d3951b42702a'
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_get_task_push_notification_config(
            params, create_server_call_context()
        )
    assert isinstance(exc_info.value.error, UnsupportedOperationError)


@pytest.mark.asyncio
async def test_get_task_push_notification_config_task_not_found():
    """Test on_get_task_push_notification_config when task is not found."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = None  # Task not found
    mock_push_store = AsyncMock(spec=PushNotificationConfigStore)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=mock_push_store,
    )
    params = GetTaskPushNotificationConfigParams(
        id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_get_task_push_notification_config(
            params, create_server_call_context()
        )

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    mock_task_store.get.assert_awaited_once_with(
        'b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_push_store.get_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_task_push_notification_config_info_not_found():
    """Test on_get_task_push_notification_config when push_config_store.get_info returns None."""
    mock_task_store = AsyncMock(spec=TaskStore)

    sample_task = create_sample_task(
        task_id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_task_store.get.return_value = sample_task

    mock_push_store = AsyncMock(spec=PushNotificationConfigStore)
    mock_push_store.get_info.return_value = None  # Info not found

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=mock_push_store,
    )
    params = GetTaskPushNotificationConfigParams(
        id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_get_task_push_notification_config(
            params, create_server_call_context()
        )

    assert isinstance(
        exc_info.value.error, InternalError
    )  # Current code raises InternalError
    mock_task_store.get.assert_awaited_once_with(
        'b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_push_store.get_info.assert_awaited_once_with(
        'b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )


@pytest.mark.asyncio
async def test_get_task_push_notification_config_info_with_config():
    """Test on_get_task_push_notification_config with valid push config id"""
    mock_task_store = AsyncMock(spec=TaskStore)

    push_store = InMemoryPushNotificationConfigStore()

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )

    set_config_params = TaskPushNotificationConfig(
        task_id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
        push_notification_config=PushNotificationConfig(
            id='81abbba0-e8eb-48e1-828b-75fd20663c34',
            url='http://1.example.com',
        ),
    )
    await request_handler.on_set_task_push_notification_config(
        set_config_params, create_server_call_context()
    )

    params = GetTaskPushNotificationConfigParams(
        id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
        push_notification_config_id='81abbba0-e8eb-48e1-828b-75fd20663c34',
    )

    result: TaskPushNotificationConfig = (
        await request_handler.on_get_task_push_notification_config(
            params, create_server_call_context()
        )
    )

    assert result is not None
    assert result.task_id == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    assert (
        result.push_notification_config.url
        == set_config_params.push_notification_config.url
    )
    assert (
        result.push_notification_config.id
        == '81abbba0-e8eb-48e1-828b-75fd20663c34'
    )


@pytest.mark.asyncio
async def test_get_task_push_notification_config_info_with_config_no_id():
    """Test on_get_task_push_notification_config with no push config id"""
    mock_task_store = AsyncMock(spec=TaskStore)

    push_store = InMemoryPushNotificationConfigStore()

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )

    set_config_params = TaskPushNotificationConfig(
        task_id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
        push_notification_config=PushNotificationConfig(
            url='http://1.example.com'
        ),
    )
    await request_handler.on_set_task_push_notification_config(
        set_config_params, create_server_call_context()
    )

    params = TaskIdParams(id='0a9970e3-0cdd-4726-899f-a1dfef92bd64')

    result: TaskPushNotificationConfig = (
        await request_handler.on_get_task_push_notification_config(
            params, create_server_call_context()
        )
    )

    assert result is not None
    assert result.task_id == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    assert (
        result.push_notification_config.url
        == set_config_params.push_notification_config.url
    )
    assert (
        result.push_notification_config.id
        == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    )


@pytest.mark.asyncio
async def test_on_resubscribe_to_task_task_not_found():
    """Test on_resubscribe_to_task when the task is not found."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = None  # Task not found

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(), task_store=mock_task_store
    )
    params = TaskIdParams(id='8c897b3f-89a8-4692-aa72-4c33367d27b8')

    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        # Need to consume the async generator to trigger the error
        async for _ in request_handler.on_resubscribe_to_task(
            params, create_server_call_context()
        ):
            pass

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    mock_task_store.get.assert_awaited_once_with(
        '8c897b3f-89a8-4692-aa72-4c33367d27b8'
    )


@pytest.mark.asyncio
async def test_on_resubscribe_to_task_queue_not_found():
    """Test on_resubscribe_to_task when the queue is not found by queue_manager.tap."""
    mock_task_store = AsyncMock(spec=TaskStore)
    sample_task = create_sample_task(
        task_id='306589b7-57f8-48a0-8b8c-d1ab5aa237e0'
    )
    mock_task_store.get.return_value = sample_task

    mock_queue_manager = AsyncMock(spec=QueueManager)
    mock_queue_manager.tap.return_value = None  # Queue not found

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        queue_manager=mock_queue_manager,
    )
    params = TaskIdParams(id='306589b7-57f8-48a0-8b8c-d1ab5aa237e0')

    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        async for _ in request_handler.on_resubscribe_to_task(
            params, create_server_call_context()
        ):
            pass

    assert isinstance(
        exc_info.value.error, TaskNotFoundError
    )  # Should be TaskNotFoundError as per spec
    mock_task_store.get.assert_awaited_once_with(
        '306589b7-57f8-48a0-8b8c-d1ab5aa237e0'
    )
    mock_queue_manager.tap.assert_awaited_once_with(
        '306589b7-57f8-48a0-8b8c-d1ab5aa237e0'
    )


@pytest.mark.asyncio
async def test_on_message_send_stream():
    request_handler = DefaultRequestHandler(
        DummyAgentExecutor(), InMemoryTaskStore()
    )
    message_params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='4a90ce5d-eda0-44be-afae-a709621eb63c',
            parts=[Part(root=TextPart(text='How are you?'))],
        ),
    )

    async def consume_stream():
        events = []
        async for event in request_handler.on_message_send_stream(
            message_params
        ):
            events.append(event)
            if len(events) >= 3:
                break  # Stop after a few events

        return events

    # Consume first 3 events from the stream and measure time
    start = time.perf_counter()
    events = await consume_stream()
    elapsed = time.perf_counter() - start

    # Assert we received events quickly
    assert len(events) == 3
    assert elapsed < 0.5

    texts = [p.root.text for e in events for p in e.status.message.parts]
    assert texts == ['Event 0', 'Event 1', 'Event 2']


@pytest.mark.asyncio
async def test_list_task_push_notification_config_no_store():
    """Test on_list_task_push_notification_config when _push_config_store is None."""
    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=AsyncMock(spec=TaskStore),
        push_config_store=None,  # Explicitly None
    )
    params = ListTaskPushNotificationConfigParams(
        id='13d5b8a8-62d7-4490-98c8-d3951b42702a'
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_list_task_push_notification_config(
            params, create_server_call_context()
        )
    assert isinstance(exc_info.value.error, UnsupportedOperationError)


@pytest.mark.asyncio
async def test_list_task_push_notification_config_task_not_found():
    """Test on_list_task_push_notification_config when task is not found."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = None  # Task not found
    mock_push_store = AsyncMock(spec=PushNotificationConfigStore)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=mock_push_store,
    )
    params = ListTaskPushNotificationConfigParams(
        id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_list_task_push_notification_config(
            params, create_server_call_context()
        )

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    mock_task_store.get.assert_awaited_once_with(
        'b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_push_store.get_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_no_task_push_notification_config_info():
    """Test on_get_task_push_notification_config when push_config_store.get_info returns []"""
    mock_task_store = AsyncMock(spec=TaskStore)

    sample_task = create_sample_task(
        task_id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_task_store.get.return_value = sample_task

    push_store = InMemoryPushNotificationConfigStore()

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )
    params = ListTaskPushNotificationConfigParams(
        id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )

    result = await request_handler.on_list_task_push_notification_config(
        params, create_server_call_context()
    )
    assert result == []


@pytest.mark.asyncio
async def test_list_task_push_notification_config_info_with_config():
    """Test on_list_task_push_notification_config with push config+id"""
    mock_task_store = AsyncMock(spec=TaskStore)

    sample_task = create_sample_task(
        task_id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_task_store.get.return_value = sample_task

    push_config1 = PushNotificationConfig(
        id='37d6af86-231f-40a5-8dba-02d2ccb4e161', url='http://example.com'
    )
    push_config2 = PushNotificationConfig(
        id='86d0a728-e0f5-4a5c-af2d-e43bd116780c', url='http://example.com'
    )

    push_store = InMemoryPushNotificationConfigStore()
    await push_store.set_info(
        '0a9970e3-0cdd-4726-899f-a1dfef92bd64', push_config1
    )
    await push_store.set_info(
        '0a9970e3-0cdd-4726-899f-a1dfef92bd64', push_config2
    )

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )
    params = ListTaskPushNotificationConfigParams(
        id='0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    )

    result: list[
        TaskPushNotificationConfig
    ] = await request_handler.on_list_task_push_notification_config(
        params, create_server_call_context()
    )

    assert len(result) == 2
    assert str(result[0].task_id) == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    assert result[0].push_notification_config == push_config1
    assert str(result[1].task_id) == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    assert result[1].push_notification_config == push_config2


@pytest.mark.asyncio
async def test_list_task_push_notification_config_info_with_config_and_no_id():
    """Test on_list_task_push_notification_config with no push config id"""
    mock_task_store = AsyncMock(spec=TaskStore)

    push_store = InMemoryPushNotificationConfigStore()

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )

    # multiple calls without config id should replace the existing
    set_config_params1 = TaskPushNotificationConfig(
        task_id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
        push_notification_config=PushNotificationConfig(
            url='http://1.example.com'
        ),
    )
    await request_handler.on_set_task_push_notification_config(
        set_config_params1, create_server_call_context()
    )

    set_config_params2 = TaskPushNotificationConfig(
        task_id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
        push_notification_config=PushNotificationConfig(
            url='http://2.example.com'
        ),
    )
    await request_handler.on_set_task_push_notification_config(
        set_config_params2, create_server_call_context()
    )

    params = ListTaskPushNotificationConfigParams(
        id='0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    )

    result: list[
        TaskPushNotificationConfig
    ] = await request_handler.on_list_task_push_notification_config(
        params, create_server_call_context()
    )

    assert len(result) == 1
    assert str(result[0].task_id) == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    assert (
        result[0].push_notification_config.url
        == set_config_params2.push_notification_config.url
    )
    assert (
        str(result[0].push_notification_config.id)
        == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    )


@pytest.mark.asyncio
async def test_delete_task_push_notification_config_no_store():
    """Test on_delete_task_push_notification_config when _push_config_store is None."""
    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=AsyncMock(spec=TaskStore),
        push_config_store=None,  # Explicitly None
    )
    params = DeleteTaskPushNotificationConfigParams(
        id='13d5b8a8-62d7-4490-98c8-d3951b42702a',
        push_notification_config_id='2fbdf017-d3d0-498f-9135-ebf8d2ae8492',
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_delete_task_push_notification_config(
            params, create_server_call_context()
        )
    assert isinstance(exc_info.value.error, UnsupportedOperationError)


@pytest.mark.asyncio
async def test_delete_task_push_notification_config_task_not_found():
    """Test on_delete_task_push_notification_config when task is not found."""
    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = None  # Task not found
    mock_push_store = AsyncMock(spec=PushNotificationConfigStore)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=mock_push_store,
    )
    params = DeleteTaskPushNotificationConfigParams(
        id='b87b95a6-cf02-4d0a-8355-eb9cf307d323',
        push_notification_config_id='2fbdf017-d3d0-498f-9135-ebf8d2ae8492',
    )
    from a2a.utils.errors import ServerError  # Local import

    with pytest.raises(ServerError) as exc_info:
        await request_handler.on_delete_task_push_notification_config(
            params, create_server_call_context()
        )

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    mock_task_store.get.assert_awaited_once_with(
        'b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_push_store.get_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_no_task_push_notification_config_info():
    """Test on_delete_task_push_notification_config without config info"""
    mock_task_store = AsyncMock(spec=TaskStore)

    sample_task = create_sample_task(
        task_id='0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    )
    mock_task_store.get.return_value = sample_task

    push_store = InMemoryPushNotificationConfigStore()
    await push_store.set_info(
        'task_2',
        PushNotificationConfig(
            id='37d6af86-231f-40a5-8dba-02d2ccb4e161', url='http://example.com'
        ),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )
    params = DeleteTaskPushNotificationConfigParams(
        id='13d5b8a8-62d7-4490-98c8-d3951b42702a',
        push_notification_config_id='fb0133f8-0cf7-4e3e-adab-d65b8df0ae22',
    )

    result = await request_handler.on_delete_task_push_notification_config(
        params, create_server_call_context()
    )
    assert result is None

    params = DeleteTaskPushNotificationConfigParams(
        id='c20360a1-201f-46d0-9e1d-5d5a6cb2601c',
        push_notification_config_id='fb0133f8-0cf7-4e3e-adab-d65b8df0ae22',
    )

    result = await request_handler.on_delete_task_push_notification_config(
        params, create_server_call_context()
    )
    assert result is None


@pytest.mark.asyncio
async def test_delete_task_push_notification_config_info_with_config():
    """Test on_list_task_push_notification_config with push config+id"""
    mock_task_store = AsyncMock(spec=TaskStore)

    sample_task = create_sample_task(
        task_id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_task_store.get.return_value = sample_task

    push_config1 = PushNotificationConfig(
        id='37d6af86-231f-40a5-8dba-02d2ccb4e161', url='http://example.com'
    )
    push_config2 = PushNotificationConfig(
        id='86d0a728-e0f5-4a5c-af2d-e43bd116780c', url='http://example.com'
    )

    push_store = InMemoryPushNotificationConfigStore()
    await push_store.set_info(
        '0a9970e3-0cdd-4726-899f-a1dfef92bd64', push_config1
    )
    await push_store.set_info(
        '0a9970e3-0cdd-4726-899f-a1dfef92bd64', push_config2
    )
    await push_store.set_info('task_2', push_config1)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )
    params = DeleteTaskPushNotificationConfigParams(
        id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
        push_notification_config_id='37d6af86-231f-40a5-8dba-02d2ccb4e161',
    )

    result1 = await request_handler.on_delete_task_push_notification_config(
        params, create_server_call_context()
    )

    assert result1 is None

    result2 = await request_handler.on_list_task_push_notification_config(
        ListTaskPushNotificationConfigParams(
            id='0a9970e3-0cdd-4726-899f-a1dfef92bd64'
        ),
        create_server_call_context(),
    )

    assert len(result2) == 1
    assert result2[0].task_id == '0a9970e3-0cdd-4726-899f-a1dfef92bd64'
    assert result2[0].push_notification_config == push_config2


@pytest.mark.asyncio
async def test_delete_task_push_notification_config_info_with_config_and_no_id():
    """Test on_list_task_push_notification_config with no push config id"""
    mock_task_store = AsyncMock(spec=TaskStore)

    sample_task = create_sample_task(
        task_id='b87b95a6-cf02-4d0a-8355-eb9cf307d323'
    )
    mock_task_store.get.return_value = sample_task

    push_config = PushNotificationConfig(url='http://example.com')

    # insertion without id should replace the existing config
    push_store = InMemoryPushNotificationConfigStore()
    await push_store.set_info(
        '0a9970e3-0cdd-4726-899f-a1dfef92bd64', push_config
    )
    await push_store.set_info(
        '0a9970e3-0cdd-4726-899f-a1dfef92bd64', push_config
    )

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        push_config_store=push_store,
    )
    params = DeleteTaskPushNotificationConfigParams(
        id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
        push_notification_config_id='0a9970e3-0cdd-4726-899f-a1dfef92bd64',
    )

    result = await request_handler.on_delete_task_push_notification_config(
        params, create_server_call_context()
    )

    assert result is None

    result2 = await request_handler.on_list_task_push_notification_config(
        ListTaskPushNotificationConfigParams(
            id='0a9970e3-0cdd-4726-899f-a1dfef92bd64'
        ),
        create_server_call_context(),
    )

    assert len(result2) == 0


TERMINAL_TASK_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_state', TERMINAL_TASK_STATES)
async def test_on_message_send_task_in_terminal_state(terminal_state):
    """Test on_message_send when task is already in a terminal state."""
    task_id = '5520dd18-ad0a-4307-a295-c4c93b543932'
    terminal_task = create_sample_task(
        task_id=task_id, status_state=terminal_state
    )

    mock_task_store = AsyncMock(spec=TaskStore)
    # The get method of TaskManager calls task_store.get.
    # We mock TaskManager.get_task which is an async method.
    # So we should patch that instead.

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(), task_store=mock_task_store
    )

    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='4d1249a9-e290-4b55-8e16-691b060bedb1',
            parts=[],
            task_id=task_id,
        )
    )

    from a2a.utils.errors import ServerError

    # Patch the TaskManager's get_task method to return our terminal task
    with patch(
        'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
        return_value=terminal_task,
    ):
        with pytest.raises(ServerError) as exc_info:
            await request_handler.on_message_send(
                params, create_server_call_context()
            )

    assert isinstance(exc_info.value.error, InvalidParamsError)
    assert exc_info.value.error.message
    assert (
        f'Task {task_id} is in terminal state: {terminal_state.value}'
        in exc_info.value.error.message
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_state', TERMINAL_TASK_STATES)
async def test_on_message_send_stream_task_in_terminal_state(terminal_state):
    """Test on_message_send_stream when task is already in a terminal state."""
    task_id = 'f7075439-b5c9-4286-a09a-a7a2d2756a03'
    terminal_task = create_sample_task(
        task_id=task_id, status_state=terminal_state
    )

    mock_task_store = AsyncMock(spec=TaskStore)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(), task_store=mock_task_store
    )

    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='4d1249a9-e290-4b55-8e16-691b060bedb1',
            parts=[],
            task_id=task_id,
        )
    )

    from a2a.utils.errors import ServerError

    with patch(
        'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
        return_value=terminal_task,
    ):
        with pytest.raises(ServerError) as exc_info:
            async for _ in request_handler.on_message_send_stream(
                params, create_server_call_context()
            ):
                pass  # pragma: no cover

    assert isinstance(exc_info.value.error, InvalidParamsError)
    assert exc_info.value.error.message
    assert (
        f'Task {task_id} is in terminal state: {terminal_state.value}'
        in exc_info.value.error.message
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_state', TERMINAL_TASK_STATES)
async def test_on_resubscribe_to_task_in_terminal_state(terminal_state):
    """Test on_resubscribe_to_task when task is in a terminal state."""
    task_id = 'a12bb281-c7ab-491f-8b3f-9913addca78a'
    terminal_task = create_sample_task(
        task_id=task_id, status_state=terminal_state
    )

    mock_task_store = AsyncMock(spec=TaskStore)
    mock_task_store.get.return_value = terminal_task

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(),
        task_store=mock_task_store,
        queue_manager=AsyncMock(spec=QueueManager),
    )
    params = TaskIdParams(id=task_id)

    from a2a.utils.errors import ServerError

    with pytest.raises(ServerError) as exc_info:
        async for _ in request_handler.on_resubscribe_to_task(
            params, create_server_call_context()
        ):
            pass  # pragma: no cover

    assert isinstance(exc_info.value.error, InvalidParamsError)
    assert exc_info.value.error.message
    assert (
        f'Task {task_id} is in terminal state: {terminal_state.value}'
        in exc_info.value.error.message
    )
    mock_task_store.get.assert_awaited_once_with(task_id)


@pytest.mark.asyncio
async def test_on_message_send_task_id_provided_but_task_not_found():
    """Test on_message_send when task_id is provided but task doesn't exist."""
    task_id = 'd95515a2-b95b-4706-a4d2-b1d2e8b068d0'
    mock_task_store = AsyncMock(spec=TaskStore)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(), task_store=mock_task_store
    )

    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='c5e7c368-8f16-4518-bd4a-d61be13858fe',
            parts=[Part(root=TextPart(text='Hello'))],
            task_id=task_id,
            context_id='06cc947f-8946-4bde-b776-165462407e57',
        )
    )

    from a2a.utils.errors import ServerError

    # Mock TaskManager.get_task to return None (task not found)
    with patch(
        'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
        return_value=None,
    ):
        with pytest.raises(ServerError) as exc_info:
            await request_handler.on_message_send(
                params, create_server_call_context()
            )

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    assert exc_info.value.error.message
    assert (
        f'Task {task_id} was specified but does not exist'
        in exc_info.value.error.message
    )


@pytest.mark.asyncio
async def test_on_message_send_stream_task_id_provided_but_task_not_found():
    """Test on_message_send_stream when task_id is provided but task doesn't exist."""
    task_id = 'e4cc2ed4-117c-4bc6-90a4-415f0a7dec5a'
    mock_task_store = AsyncMock(spec=TaskStore)

    request_handler = DefaultRequestHandler(
        agent_executor=DummyAgentExecutor(), task_store=mock_task_store
    )

    params = MessageSendParams(
        message=Message(
            role=Role.user,
            message_id='e3d82e74-63ac-4603-b54a-fce0bb01c447',
            parts=[Part(root=TextPart(text='Hello'))],
            task_id=task_id,
            context_id='06cc947f-8946-4bde-b776-165462407e57',
        )
    )

    from a2a.utils.errors import ServerError

    # Mock TaskManager.get_task to return None (task not found)
    with patch(
        'a2a.server.request_handlers.default_request_handler.TaskManager.get_task',
        return_value=None,
    ):
        with pytest.raises(ServerError) as exc_info:
            # Need to consume the async generator to trigger the error
            async for _ in request_handler.on_message_send_stream(
                params, create_server_call_context()
            ):
                pass

    assert isinstance(exc_info.value.error, TaskNotFoundError)
    assert exc_info.value.error.message
    assert (
        f'Task {task_id} was specified but does not exist'
        in exc_info.value.error.message
    )
