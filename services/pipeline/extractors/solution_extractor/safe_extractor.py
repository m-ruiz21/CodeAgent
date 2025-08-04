from typing import Any, List

from typing import Any, Dict, List, Sequence
from llama_index.core.schema import BaseNode
from llama_index.core.extractors.interface import BaseExtractor

from utils.retries_wrapper import retries_wrapper

class SafeExtractor(BaseExtractor):
    """
    Solution extractor. Node-level extractor. Extracts
    `solution` metadata field. 
    """

    _base_extractor: BaseExtractor
    def __init__(
        self,
        extractor: BaseExtractor,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            **kwargs,
        )
        self._base_extractor = extractor
    
    @classmethod
    def class_name(cls) -> str:
        return "SafeExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        try:
            return await retries_wrapper(
                fun=lambda: self._base_extractor.aextract(nodes),
                retries=3,
                desc="Extracting solution names"
            )
        except Exception as e:
            failed_paths = [node.metadata.get("file_path", "") for node in nodes if not node.metadata.get("file_path", "").startswith("Solutions/")]
            print(f"Error occurred while extracting solutions from nodes {failed_paths}: {e}")
            return []