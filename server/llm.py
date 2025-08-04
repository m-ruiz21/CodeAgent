from fastmcp import FastMCP, Context
from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings

from pydantic_ai.models.mcp_sampling import MCPSamplingModel


class SamplingLLM(CustomLLM):
    """
    A model that uses MCP Sampling.

    [MCP Sampling](https://modelcontextprotocol.io/docs/concepts/sampling)
    allows an MCP server to make requests to a model by calling back to the MCP client that connected to it.
    """
    dummy_response: str = "My response"

    def __init__(
        self,
        mcp_context: Context,
        context_window: int = 3900,
        num_output: int = 256,
        model_name: str = "Sampling Model",
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        self.mcp_context = mcp_context
        super().__init__(
            context_window=context_window,
            num_output=num_output,
            model_name=model_name,
            callback_manager=callback_manager,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    async def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = await self.mcp_context.sample(prompt, )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    async def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError("MCP Sampling does not support streaming.")