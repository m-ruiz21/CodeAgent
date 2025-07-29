import json
from typing import Any, List, Optional
from tqdm import tqdm

from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core import Settings
from llama_index.core.llms import LLM

from services.pipeline.context_enrichment.context import Context
from services.pipeline.context_enrichment.prompt import get_prompt
from services.cache.doc_service import get_doc_service

class ContextEnricher(TransformComponent):
    def __init__(self, llm: Optional[LLM] = None) -> None:
        _llm = llm or Settings.llm
        self._doc_service = get_doc_service()
        self._sllm = _llm.as_structured_llm(Context) 

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        if kwargs.get("show_progress", True):
            nodes = tqdm(nodes, desc="Enriching Code Chunks with Context", unit="chunk")

        out_nodes = []
        for node in nodes:
            file_path = node.metadata.get("file_path", "")
            
            if file_path.endswith(".md"):
                out_nodes.append(node)
                continue

            enriched_node = self._enrich_node(node)
            file_path = node.metadata.get("file_path", "")
            solution =  file_path.split('/')[1]
            enriched_node.metadata.update({
                "solution": solution,
            })

            out_nodes.append(enriched_node)

        return out_nodes

    def _set_prompt(self, node: BaseNode) -> str:
        """Set the prompt for the LLM based on the node."""
        file_path = node.metadata.get("file_path", "")

        file_content = self._doc_service.get_content(file_path)
        prompt = get_prompt(node.text, file_content)
        return prompt

    def _enrich_node(self, node: BaseNode) -> BaseNode:
        """Enrich a single node with additional context."""
        prompt = self._set_prompt(node)
        response = self._sllm.complete(prompt)
        context: Context = response.raw

        new_text = context.prepend + node.text + context.postpend

        node.text = new_text
        node.metadata.update({
            "entities": json.dumps(context.entities),
            "local_dependencies": json.dumps(context.local_dependencies),
            "imports": json.dumps(context.imports),
        })

        return node
