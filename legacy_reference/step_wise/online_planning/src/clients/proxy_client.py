"""
OpenAI-compatible proxy client for model inference.

This module provides a client for communicating with OpenAI-compatible APIs,
including support for custom MAAS models and various providers.
"""

import os
import base64
from typing import Dict, List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
import tiktoken


VLLM_CONFIGS = {
    "qwen3vl-235B": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1",
    },
    "qwen3vl-30B": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1",
    },
    "internvl3-5-241B": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1",
    },
    "internvl3-5-30B": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1",
    },
    "internvl3-5-38B": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1",
    },
    "qwen3vl-32B": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1",
    },
}


class ProxyClient:
    """Simplified OpenAI-compatible proxy client."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model

        # If explicit configuration is provided, use it directly
        if base_url:
            self.client = OpenAI(
                api_key=api_key or "EMPTY",
                base_url=base_url,
                max_retries=3,
                timeout=300.0,
            )
            self.is_maas_model = False
            self.is_vllm_model = True  # Assume custom endpoints are vLLM-like
            self.vllm_model_name = model
            print(f"✓ Initialized custom client: {base_url} with model {model}")
            return

        # Check if this is a vLLM model (Qwen3-VL or InternVL3, case-insensitive)
        model_lower = model.lower()
        is_vllm_target = (
            model_lower.startswith("qwen3vl-30b")
            or model_lower.startswith("qwen3vl-32b")
            or model_lower.startswith("internvl3")
        )

        if is_vllm_target:
            # Use vLLM configuration

            # Check for specific configuration first
            if model in VLLM_CONFIGS:
                config = VLLM_CONFIGS[model]
                base_url = config["base_url"]
                api_key = config["api_key"]
            else:
                # Fallback to env vars or default
                # vLLM base URL - can be overridden by environment variable
                base_url = os.getenv("VLLM_BASE_URL", "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1")
                api_key = os.getenv("VLLM_API_KEY", "EMPTY")

            # vLLM model name - can be overridden by environment variable, default to the model name itself
            vllm_model_name = os.getenv("VLLM_MODEL_NAME", model)

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=3,
                timeout=300.0,  # Longer timeout for vLLM
            )
            self.is_maas_model = False
            self.is_vllm_model = True
            self.vllm_model_name = (
                vllm_model_name  # Store the actual vLLM model name to use
            )
            print(
                f"✓ Initialized vLLM client: {base_url} with model {model} (vLLM model: {vllm_model_name})"
            )

        # Check if this is a custom MAAS model
        elif model == "qw3_2050_agent_planner_a3b_ep3_1010":
            # Use custom MAAS configuration
            api_key = "YOUR_API_KEY"
            base_url = "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.is_maas_model = True
            self.is_vllm_model = False
            print(f"✓ Initialized MAAS client: {base_url} with model {model}")

        # Check if this is the Intern241b model
        elif model == "internvl3-5-241B":
            # Use InternAI configuration
            api_key = os.getenv("INTERN_API_KEY")
            if not api_key:
                raise ValueError("INTERN_API_KEY environment variable not set")
            base_url = "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self.is_maas_model = False
            self.is_vllm_model = False
            print(f"✓ Initialized InternAI client: {base_url} with model {model}")

        else:
            # Standard proxy configuration
            self.is_maas_model = False
            self.is_vllm_model = False
            provider = self._get_provider_from_model(model)

            # Get API key based on provider
            api_key_var = f"{provider.upper()}_PROXY_API_KEY"
            api_key_temp = os.getenv(api_key_var)
            base_url_temp = os.getenv("PROXY_BASE_URL")

            if not api_key_temp:
                raise ValueError(f"{api_key_var} environment variable not set")
            if not base_url_temp:
                raise ValueError("PROXY_BASE_URL environment variable not set")

            api_key = api_key_temp
            base_url = base_url_temp

            # Add retries and timeout to handle unstable proxy connections
            self.client = OpenAI(
                api_key=api_key, base_url=base_url, max_retries=5, timeout=120.0
            )
            print(
                f"✓ Initialized proxy client: {base_url} with model {model} (provider: {provider}, API key: {api_key_var})"
            )

    def _get_provider_from_model(self, model: str) -> str:
        """Determine the provider from model name using simple string matching."""
        model_lower = model.lower()

        # Simple string matching
        if "gpt" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "gemini"
        elif "claude" in model_lower:
            return "claude"
        elif "grok" in model_lower:
            return "grok"
        elif "qwen" in model_lower:
            return "qwen"
        elif "intern" in model_lower:
            return "intern"
        else:
            # Default to openai if no match found
            print(
                f"⚠️  Warning: Could not determine provider for model '{model}', defaulting to 'openai'"
            )
            return "openai"

    def _encode_image(self, image_path: str) -> str:
        # """Encode image to base64 with 50% resizing."""
        # try:
        #     with Image.open(image_path) as img:
        #         # Calculate new dimensions (50% of original)
        #         new_width = int(img.width * 0.5)
        #         new_height = int(img.height * 0.5)

        #         # Resize image
        #         resized_img = img.resize(
        #             (new_width, new_height), Image.Resampling.LANCZOS
        #         )

        #         # Save to buffer
        #         buffer = io.BytesIO()
        #         # Preserve format if possible, default to JPEG if not available
        #         fmt = img.format if img.format else "JPEG"
        #         resized_img.save(buffer, format=fmt)

        #         # Encode
        #         return base64.b64encode(buffer.getvalue()).decode("utf-8")
        # except Exception as e:
        #     print(f"Error resizing image {image_path}: {e}. Falling back to original.")
        #     # Fallback to original if resizing fails
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _read_text_file(self, file_path: str) -> str:
        """Read text file content."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback if tiktoken fails or encoding not found
            return len(text) // 4

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 64000,
        files: Optional[List[Dict]] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate response from model using streaming.

        Args:
            prompt: Text prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            files: Optional list of files, each with 'type' and 'path' keys
            temperature: Sampling temperature (0.0 = deterministic, default)

        Returns:
            Generated text response
        """
        messages = []

        # Token stats container
        stats = {"system_prompt": 0, "files": [], "main_prompt": 0, "total": 0}

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            count = self._count_tokens(system_prompt)
            stats["system_prompt"] = count
            stats["total"] += count

        # Build user message content with files
        user_content = []

        # Add files if provided
        if files:
            print(f"  📎 Processing {len(files)} file(s)...")
            for file_info in files:
                file_type = file_info.get("type", "text")
                file_path = file_info.get("path", "")

                if not os.path.exists(file_path):
                    print(f"     ⚠️  File not found: {file_path}")
                    continue

                if file_type == "image":
                    # Add image as base64
                    try:
                        image_data = self._encode_image(file_path)
                        # Determine image type from extension
                        ext = os.path.splitext(file_path)[1].lower()
                        mime_type = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".gif": "image/gif",
                            ".webp": "image/webp",
                        }.get(ext, "image/jpeg")

                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            }
                        )

                        # Estimate image tokens (approx 1280 for high-res)
                        # This is an estimation to avoid context length errors
                        img_tokens = 1280
                        stats["files"].append(
                            {
                                "name": os.path.basename(file_path),
                                "tokens": img_tokens,
                                "type": "image",
                            }
                        )
                        stats["total"] += img_tokens

                        print(
                            f"     ✅ Added image: {os.path.basename(file_path)} (~{img_tokens} tokens)"
                        )
                    except Exception as e:
                        print(f"     ❌ Failed to encode image {file_path}: {e}")

                elif file_type == "text":
                    # Add text file content to prompt
                    try:
                        text_content = self._read_text_file(file_path)
                        user_content.append(
                            {
                                "type": "text",
                                "text": f"**File: {os.path.basename(file_path)}**\n```\n{text_content}\n```\n",
                            }
                        )
                        print(f"     ✅ Added text file: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"     ❌ Failed to read text file {file_path}: {e}")

                    if "text_content" in locals():
                        count = self._count_tokens(text_content)
                        stats["files"].append(
                            {
                                "name": os.path.basename(file_path),
                                "tokens": count,
                                "type": "text",
                            }
                        )
                        stats["total"] += count

        # Add main prompt
        user_content.append({"type": "text", "text": prompt})

        count = self._count_tokens(prompt)
        stats["main_prompt"] = count
        stats["total"] += count

        # Print stats
        print("\n📊 Token Statistics:")
        print(f"  - System Prompt: {stats['system_prompt']}")
        for f in stats["files"]:
            type_str = f" [{f['type']}]" if "type" in f else ""
            print(f"  - File ({f['name']}){type_str}: {f['tokens']}")
        print(f"  - Main Prompt: {stats['main_prompt']}")
        print(f"  - Total (Estimated): {stats['total']}")

        messages.append({"role": "user", "content": user_content})  # type: ignore

        # For vLLM model, use streaming mode
        if hasattr(self, "is_vllm_model") and self.is_vllm_model:
            print("  🔄 Streaming response from vLLM...", flush=True)
            full_response = ""

            try:
                # Use the vLLM model name instead of the original model name
                stream = self.client.chat.completions.create(
                    model=self.vllm_model_name,  # Use the actual vLLM model name
                    messages=messages,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=42,
                    stream=True,  # Use streaming for vLLM
                )

                # Type hint for the stream iterator
                chunk: ChatCompletionChunk
                for chunk in stream:  # type: ignore
                    if (
                        hasattr(chunk, "choices")
                        and chunk.choices
                        and len(chunk.choices) > 0
                    ):
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content is not None:
                            content = delta.content
                            # Ensure content is string
                            if isinstance(content, bytes):
                                content = content.decode("utf-8")
                            full_response += str(content)
                            # Print progress indicator every 100 characters
                            if len(full_response) % 100 == 0:
                                print(".", end="", flush=True)

                if full_response:
                    print()  # New line after progress dots

                return full_response

            except Exception as e:
                print(f"\n✗ vLLM streaming error: {e}")
                raise

        # For MAAS model, use non-streaming mode
        elif self.is_maas_model:
            print("  🔄 Processing request (non-streaming)...", end="", flush=True)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=42,
                    stream=False,
                )
                full_response = response.choices[0].message.content or ""
                print(" Done!")
                return full_response
            except Exception as e:
                print(f"\n✗ Request error: {e}")
                raise

        # Use streaming to avoid timeout for long requests (for standard models)
        print("  🔄 Streaming response...", flush=True)
        full_response = ""

        # Handle model name mapping
        model_to_use = self.model
        if self.model == "anthropic/claude-sonnet-4.5":
            model_to_use = "claude-sonnet-4-5-20250929"
        if self.model == "qwen3vl-235B":
            model_to_use = "qwen3-vl-235b-a22b-instruct"
        if self.model == "internvl3-5-241B":
            model_to_use = "internvl3.5-241b-a28b"
        if self.model == "gemini-3-pro-preview":
            model_to_use = "gpt-5"
        try:
            print(f"use {model_to_use}!!!!!!!!!!!!!!!!!!1")
            stream = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                seed=42,
                stream=True,
            )

            # Type hint for the stream iterator
            chunk: ChatCompletionChunk
            for chunk in stream:  # type: ignore
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and len(chunk.choices) > 0
                ):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content is not None:
                        content = delta.content
                        # Ensure content is string
                        if isinstance(content, bytes):
                            content = content.decode("utf-8")
                        full_response += str(content)
                        # Print progress indicator every 100 characters
                        if len(full_response) % 100 == 0:
                            print(".", end="", flush=True)

            if full_response:
                print()  # New line after progress dots

            return full_response

        except Exception as e:
            print(f"\n✗ Streaming error: {e}")
            raise
