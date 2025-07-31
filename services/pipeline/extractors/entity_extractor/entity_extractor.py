from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.llms import LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core import Settings
from pydantic import Field, SerializeAsAny
from llama_index.core.prompts import PromptTemplate

DEFAULT_ENTITY_EXTRACT_TEMPLATE = """\
{context_str}. 

Extract the most relevant entities, such as classes, functions, and variables, from the above context. Return a comma-separated list of entities.
If the variable comes with a hardcoded value, include the value in the entity name. For example, if the variable is `url = google.com`, return `url=google.com` as an entity.

Guidelines:
1. Ignore throwaway locals (loop vars, temp counters, etc.).
2. Collapse overloaded/templated variants into one logical entity.
3. Deduplicate fully-qualified vs. imported aliases.
4. For constants, include RHS if it’s a literal; otherwise null.
5. Preserve order of appearance.
6. Return a comma-separated list of entities.
"""

class EntityExtractor(BaseExtractor):
    """
    Entity extractor. Node-level extractor. Extracts
    `excerpt_entities` metadata field.

    Args:
        llm (Optional[LLM]): LLM
        prompt_template (str): template for keyword extraction

    """

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for generation.")

    prompt_template: str = Field(
        default=DEFAULT_ENTITY_EXTRACT_TEMPLATE,
        description="Prompt template to use when generating entities.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        prompt_template: str = DEFAULT_ENTITY_EXTRACT_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            llm=llm or Settings.llm,
            prompt_template=prompt_template,
            num_workers=num_workers,
            **kwargs,
        )


    @classmethod
    def class_name(cls) -> str:
        return "EntityExtractor"

    async def _aextract_entities_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract entities from a node and return it's metadata dict."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}

        context_str = node.get_content(metadata_mode=self.metadata_mode)

        entities = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template),
            context_str=context_str,
        )

        return {"excerpt_entities": entities.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        entity_jobs = []
        for node in nodes:
            entity_jobs.append(self._aextract_entities_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            entity_jobs, show_progress=self.show_progress, workers=self.num_workers, desc="Extracting entities"
        )

        return metadata_list