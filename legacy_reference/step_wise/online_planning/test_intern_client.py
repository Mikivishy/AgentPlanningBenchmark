import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from eval.online_planning.src.clients.proxy_client import ProxyClient


class TestInternClient(unittest.TestCase):
    @patch("eval.online_planning.src.clients.proxy_client.OpenAI")
    def test_intern_client_initialization(self, mock_openai):
        # Mock environment variable
        with patch.dict(os.environ, {"INTERN_API_KEY": "test_key"}):
            client = ProxyClient(model="intern241b")

            # Verify OpenAI initialization
            mock_openai.assert_called_with(
                api_key="YOUR_API_KEY", base_url="https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
            )

            # Verify client properties
            self.assertFalse(client.is_maas_model)
            self.assertFalse(client.is_vllm_model)

            # Verify generation call
            mock_chat = mock_openai.return_value.chat.completions.create
            client.generate(prompt="hello")

            # Verify that the correct model name is passed to the API
            call_args = mock_chat.call_args
            self.assertEqual(call_args.kwargs["model"], "intern-latest")


if __name__ == "__main__":
    unittest.main()
