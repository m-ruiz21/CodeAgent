import json
from typing import Any, List, Optional
from tqdm import tqdm

from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core import Settings
from llama_index.core.llms import LLM

from services.pipeline.context_enrichment.context import Context
from services.pipeline.context_enrichment.prompt import get_prompt
from services.cache.doc_service import get_doc_service

class SolutionAdder(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        show_progress = kwargs.get("show_progress", True)

        if show_progress:
            nodes_progress = tqdm(nodes, desc="Populating Solution Metadata", unit="solution")

        out_nodes = []
        for node in nodes_progress:
            file_path = node.metadata.get("file_path", "")
            solution =  file_path.split('/')[1]
            node.metadata.update({
                "solution": solution,
            })
            
            if show_progress:
                nodes_progress.set_description(f"Populating Solution Metadata for {solution}")

            out_nodes.append(node)

        return out_nodes