import unittest

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from a2a.server.tasks.base_push_notification_sender import (
    BasePushNotificationSender,
)
from a2a.types import (
    PushNotificationConfig,
    Task,
    TaskState,
    TaskStatus,
)


def create_sample_task(
    task_id='eede470e-ae8f-4910-ba05-085d45dc43c6',
    status_state=TaskState.completed,
):
    return Task(
        id=task_id,
        context_id='a2e44180-c4f5-4bdb-9c57-5151b145a0cd',
        status=TaskStatus(state=status_state),
    )


def create_sample_push_config(
    url='http://example.com/callback', config_id='e40b9db6-fdb2-4712-8d56-2ff86de9038f', token=None
):
    return PushNotificationConfig(id=config_id, url=url, token=token)


class TestBasePushNotificationSender(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
        self.mock_config_store = AsyncMock()
        self.sender = BasePushNotificationSender(
            httpx_client=self.mock_httpx_client,
            config_store=self.mock_config_store,
        )

    def test_constructor_stores_client_and_config_store(self):
        self.assertEqual(self.sender._client, self.mock_httpx_client)
        self.assertEqual(self.sender._config_store, self.mock_config_store)

    async def test_send_notification_success(self):
        task_id = '54a31351-1de9-4dd1-8e57-64a1ff99b2b1'
        task_data = create_sample_task(task_id=task_id)
        config = create_sample_push_config(url='http://notify.me/here')
        self.mock_config_store.get_info.return_value = [config]

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        self.mock_httpx_client.post.return_value = mock_response

        await self.sender.send_notification(task_data)

        self.mock_config_store.get_info.assert_awaited_once_with

        # assert httpx_client post method got invoked with right parameters
        self.mock_httpx_client.post.assert_awaited_once_with(
            config.url,
            json=task_data.model_dump(mode='json', exclude_none=True),
            headers=None,
        )
        mock_response.raise_for_status.assert_called_once()

    async def test_send_notification_with_token_success(self):
        task_id = '54a31351-1de9-4dd1-8e57-64a1ff99b2b1'
        task_data = create_sample_task(task_id=task_id)
        config = create_sample_push_config(
            url='http://notify.me/here', token='unique_token'
        )
        self.mock_config_store.get_info.return_value = [config]

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        self.mock_httpx_client.post.return_value = mock_response

        await self.sender.send_notification(task_data)

        self.mock_config_store.get_info.assert_awaited_once_with

        # assert httpx_client post method got invoked with right parameters
        self.mock_httpx_client.post.assert_awaited_once_with(
            config.url,
            json=task_data.model_dump(mode='json', exclude_none=True),
            headers={'X-A2A-Notification-Token': 'unique_token'},
        )
        mock_response.raise_for_status.assert_called_once()

    async def test_send_notification_no_config(self):
        task_id = '17bf28b3-2381-472c-9fab-7d1962f630ab'
        task_data = create_sample_task(task_id=task_id)
        self.mock_config_store.get_info.return_value = []

        await self.sender.send_notification(task_data)

        self.mock_config_store.get_info.assert_awaited_once_with(task_id)
        self.mock_httpx_client.post.assert_not_called()

    @patch('a2a.server.tasks.base_push_notification_sender.logger')
    async def test_send_notification_http_status_error(
        self, mock_logger: MagicMock
    ):
        task_id = '65dad88d-500a-4629-a6ff-de7cc8d535ca'
        task_data = create_sample_task(task_id=task_id)
        config = create_sample_push_config(url='http://notify.me/http_error')
        self.mock_config_store.get_info.return_value = [config]

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.text = 'Not Found'
        http_error = httpx.HTTPStatusError(
            'Not Found', request=MagicMock(), response=mock_response
        )
        self.mock_httpx_client.post.side_effect = http_error

        await self.sender.send_notification(task_data)

        self.mock_config_store.get_info.assert_awaited_once_with(task_id)
        self.mock_httpx_client.post.assert_awaited_once_with(
            config.url,
            json=task_data.model_dump(mode='json', exclude_none=True),
            headers=None,
        )
        mock_logger.error.assert_called_once()

    async def test_send_notification_multiple_configs(self):
        task_id = '816b4b79-1b61-4f11-9a29-edfc32bc5a45'
        task_data = create_sample_task(task_id=task_id)
        config1 = create_sample_push_config(
            url='http://notify.me/cfg1', config_id='e40b9db6-fdb2-4712-8d56-2ff86de9038f'
        )
        config2 = create_sample_push_config(
            url='http://notify.me/cfg2', config_id='cfg2'
        )
        self.mock_config_store.get_info.return_value = [config1, config2]

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        self.mock_httpx_client.post.return_value = mock_response

        await self.sender.send_notification(task_data)

        self.mock_config_store.get_info.assert_awaited_once_with(task_id)
        self.assertEqual(self.mock_httpx_client.post.call_count, 2)

        # Check calls for config1
        self.mock_httpx_client.post.assert_any_call(
            config1.url,
            json=task_data.model_dump(mode='json', exclude_none=True),
            headers=None,
        )
        # Check calls for config2
        self.mock_httpx_client.post.assert_any_call(
            config2.url,
            json=task_data.model_dump(mode='json', exclude_none=True),
            headers=None,
        )
        mock_response.raise_for_status.call_count = 2
