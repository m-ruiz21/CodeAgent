from typing import Any, List
from tqdm import tqdm
from llama_index.core.schema import TransformComponent, BaseNode, TextNode
import time
start_time = time.time()

from services.pipeline.code_splitter.registry import CodeSplitterRegistry

class CodeSplitter(TransformComponent):
    def __init__(self, splitter_registry: CodeSplitterRegistry):
        self._registry = splitter_registry

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        if kwargs.get("show_progress", True):
            nodes = tqdm(nodes, desc="Splitting code files by path", unit="file")

        out_nodes = []
        for node in nodes:
            split_nodes = self._split_node(node)
            out_nodes.extend(split_nodes)

        return out_nodes
    
    def _split_node(self, node: BaseNode) -> List[BaseNode]:
        """Split a single node based on its file path."""
        file_path = node.metadata.get("file_path", "")
        if not self._registry.is_supported(node.metadata.get("file_path", "")):
            return []
        
        splitter = self._registry.get_splitter(file_path)

        split_node_text = splitter.split_text(node.text)
        return [TextNode(text=text, metadata={**node.metadata, "chunk": i}) for i, text in enumerate(split_node_text)]