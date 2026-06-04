import time
import unittest
from unittest.mock import patch, MagicMock

# Import the service
from services.webhook_service import WebhookDispatcher, WebhookEvent

class TestWebhookService(unittest.TestCase):
    def setUp(self):
        # Reset the singleton for testing
        WebhookDispatcher._instance = None
        self.dispatcher = WebhookDispatcher()
        
    @patch('requests.post')
    @patch('integration_config.add_log')
    def test_successful_dispatch(self, mock_add_log, mock_post):
        # Setup mock response
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.text = "Success"
        mock_post.return_value = mock_resp
        
        self.dispatcher.enqueue(
            url="http://test.com",
            payload={"data": 1},
            files=None,
            timeout=1.0,
            class_name="person",
            event_type="in"
        )
        
        # Wait for worker thread to process
        time.sleep(0.5)
        
        mock_post.assert_called_once()
        mock_add_log.assert_called_with("person", "in", "success", "Status 200")

    @patch('requests.post')
    @patch('integration_config.add_log')
    def test_rate_limiting(self, mock_add_log, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        
        # Mock time.sleep inside the dispatcher to actually measure wait time
        # without real sleeps interfering too much, or just measure real elapsed time
        start = time.time()
        self.dispatcher.enqueue("http://test.com/1", {}, None, 1.0, "p", "in")
        self.dispatcher.enqueue("http://test.com/2", {}, None, 1.0, "p", "in")
        
        # Two requests should take at least 1 * 0.3 = 0.3s (first is instant, second waits 300ms)
        # We wait 1.0 seconds to ensure the queue processes
        time.sleep(1.0)
        
        self.assertEqual(mock_post.call_count, 2)
        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.3)

    @patch('requests.post')
    @patch('integration_config.add_log')
    def test_retry_mechanism(self, mock_add_log, mock_post):
        # We want to test that a failed request gets put back into the queue 
        # with execute_at + 10s and retry_count increments.
        
        # Temporarily shorten delays for fast unit testing
        self.dispatcher.RETRY_DELAY = 0.1
        self.dispatcher.RATE_LIMIT_DELAY = 0.05
        
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Error"
        mock_post.return_value = mock_resp
        
        self.dispatcher.enqueue("http://test.com/fail", {}, None, 1.0, "p", "in")
        
        # Wait enough time for 3 retries (0.1s delay each + 0.05s rate limit = ~0.6s max)
        time.sleep(1.2)
        
        # Initial request + 3 retries = 4 calls total
        self.assertEqual(mock_post.call_count, 4)
        
        # Assert it gave up
        # The logs should have 3 retries and 1 permanent failure.
        log_calls = mock_add_log.call_args_list
        self.assertEqual(len(log_calls), 4)
        
        last_log = log_calls[-1][0]
        self.assertEqual(last_log[2], "error")
        self.assertIn("Permanently failed after 3 retries", last_log[3])

if __name__ == '__main__':
    unittest.main()
