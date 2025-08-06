from datetime import datetime
import math
import os
import pickle
import nltk
from typing import Iterator, List, TypeVar, Sequence
from dotenv import load_dotenv

from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import TokenTextSplitter

from services.github.utils.path_filter import DirectoryFilter, FileFilter, FilterType
from services.pipeline.extractors.solution_extractor.safe_extractor import SafeExtractor
from services.pipeline.splitters.code_splitter.code_splitter import CodeSplitter, CodeSplitterRegistry
from services.github.github_loader import GithubReader
from services.pipeline.extractors.solution_extractor.solution_extractor import SolutionExtractor
from services.pipeline.splitters.identity_splitter.identity_splitter import IdentitySplitter

load_dotenv()
github_key = os.getenv("GITHUB_KEY")

SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. \

In your summary, make sure to:
1. Mention the main entities and their meaning relative to the code and general Solution
2. Provide a concise summary that captures the essence of the code snippet and its relationship to the surrounding context
3. Highlight any important functions, classes, or methods and their purposes.
4. make sure to consider its role in the overall Azure Sentinel Connector, does it:
    a. handle data ingestion
    b. authenticate
    c. define the ui?
    d. provide any details about the general usage and purpose?

Summary: """

T = TypeVar("T")
def batched(seq: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield fixed‑size chunks from seq (last chunk may be smaller)."""
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])

def run_pipeline(url: str, branch: str, language_model: LLM, embed_model: BaseEmbedding, vector_store: RedisVectorStore, github_key: str) -> None:

    cache = IngestionCache(
            cache=RedisKVStore.from_host_and_port("localhost", 6379),
            collection="redis_cache"
        )

    doc_store = RedisDocumentStore.from_host_and_port("localhost", 6379, namespace="document_store")
    
    docs = GithubReader(token=str(github_key), url=url, branch=branch, parse=False,
                        file_filters=[
                            FileFilter(regex=r"^Solutions/[^/]+/Data Connectors/.*", filter_type=FilterType.INCLUDE),
                            FileFilter(regex=r"\.(zip|tar|png|jpg|jpeg|svg)$", filter_type=FilterType.EXCLUDE),
                        ],
                        directory_filters=[
                            DirectoryFilter(regex=r"^Solutions", filter_type=FilterType.INCLUDE),
                            DirectoryFilter(regex=r"^Solutions/[^/]+/(?!Data Connectors).*", filter_type=FilterType.EXCLUDE),
                            DirectoryFilter(regex=r"(?:\.python_packages|__pycache__|/lib/)", filter_type=FilterType.EXCLUDE),
                        ]).load_data()

    splitter_registry = CodeSplitterRegistry(chunk_lines=100, chunk_lines_overlap=25, max_chars=10000)
    token_truncator = TokenTextSplitter(    # enforce ada token limit
        chunk_size=7800,                    # safe headroom for ~8k-token ada
        chunk_overlap=300,
        backup_separators=["\n\n", "\n", " "],
    )

    pipeline = IngestionPipeline(
        transformations=[
            CodeSplitter(splitter_registry=splitter_registry), 
            SolutionExtractor(), 
            SafeExtractor(
                SummaryExtractor(llm=language_model or Settings.llm, prompt_template=SUMMARY_EXTRACT_TEMPLATE, num_workers=3)
            ),
            SafeExtractor(
                QuestionsAnsweredExtractor(llm=language_model or Settings.llm, questions=3, num_workers=3)
            ),
            token_truncator, # truncate to fit in ada token limit
            embed_model or Settings.embed_model
        ],
        docstore=doc_store,
        vector_store=vector_store,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )

    print(f"Loaded {len(docs)} documents from GitHub repository.")
  
    BATCH_SIZE = 100
    total_nodes = 0
    print(f"Processing {len(docs)} documents in batches of {BATCH_SIZE}...")
    for batch_num, doc_batch in enumerate(batched(docs, BATCH_SIZE), 1):
        start_time = datetime.now()
        print(f"Processing batch {batch_num} / {len(docs) // BATCH_SIZE + 1} with {len(doc_batch)} documents")
        nodes = pipeline.run(documents=doc_batch, show_progress=True)
        total_nodes += len(nodes)
        end_time = datetime.now()

        print(f"Batch {batch_num} processed in {(end_time - start_time).total_seconds():.2f} seconds.")
        print(f"-> {len(nodes)} nodes ingested (running total: {total_nodes})")

    print(f"Done. Ingested {total_nodes} nodes.")
