from typing import Any, List

from typing import Any, Dict, List, Sequence
from llama_index.core.schema import BaseNode
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs

class SolutionExtractor(BaseExtractor):
    """
    Solution extractor. Node-level extractor. Extracts
    `solution` metadata field. 
    """
    def __init__(
        self,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            num_workers=num_workers,
            **kwargs,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "EntityExtractor"

    async def _aextract_solution_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract solution from a node and return it's metadata dict."""
        file_path = node.metadata.get("file_path", "")
        if not file_path.startswith("Solutions/") or len(file_path.split('/')) < 2:
            return {}
        
        solution =  file_path.split('/')[1]
        return {"solution": solution}

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        jobs = []
        for node in nodes:
            jobs.append(self._aextract_solution_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            jobs, show_progress=self.show_progress, workers=self.num_workers, desc="Extracting solution names"
        )

        return metadata_list