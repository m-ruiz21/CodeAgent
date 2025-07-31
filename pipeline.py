from datetime import datetime
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

from services.github.utils.path_filter import DirectoryFilter, FileFilter, FilterType
from services.pipeline.context_enrichment.context_enricher import ContextEnricher
from services.pipeline.splitters.code_splitter.code_splitter import CodeSplitter, CodeSplitterRegistry
from services.github.github_loader import GithubReader
from services.cache.doc_service import DocService, set_doc_service
from services.pipeline.extractors.entity_extractor.entity_extractor import EntityExtractor
from services.pipeline.extractors.solution_extractor.solution_extractor import SolutionExtractor
from services.pipeline.splitters.identity_splitter.identity_splitter import IdentitySplitter

load_dotenv()
github_key = os.getenv("GITHUB_KEY")

SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. \

In your summary, make sure to:
1. Mention the main entities and their roles in the code. 
2. Provide a concise summary that captures the essence of the code snippet and its relationship to the surrounding context.

Summary: """

T = TypeVar("T")
def batched(seq: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield fixed‑size chunks from seq (last chunk may be smaller)."""
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])

def run_pipeline(url: str, branch: str, language_model: LLM, embed_model: BaseEmbedding, vector_store: RedisVectorStore, github_key: str) -> List[BaseNode]:

    cache = IngestionCache(
            cache=RedisKVStore.from_host_and_port("localhost", 6379),
            collection="redis_cache"
        )

    doc_store = RedisDocumentStore.from_host_and_port("localhost", 6379, namespace="document_store")
    
    vector_store.delete_index()
    vector_store.create_index()
    cache.clear()

    if os.path.exists("docs.pkl"):
        print("using cached docs.pkl")
        with open("docs.pkl", "rb") as f:
            docs = pickle.load(f)
    else:
        docs = GithubReader(token=str(github_key), url=url, branch=branch, parse=False,
                            file_filters=[
                                FileFilter(regex=r"^Solutions/[^/]+/Data Connectors/.*", filter_type=FilterType.INCLUDE),
                                FileFilter(regex=r"\.zip$", filter_type=FilterType.EXCLUDE),
                            ],
                            directory_filters=[
                                DirectoryFilter(regex=r"^Solutions", filter_type=FilterType.INCLUDE),
                                DirectoryFilter(regex=r"^Solutions/[^/]+/(?!Data Connectors).*", filter_type=FilterType.EXCLUDE),
                            ]).load_data()
        
        with open("docs.pkl", "wb") as f:
            pickle.dump(docs, f)

    # docs = GithubReader(token=str(github_key), url=url, branch=branch, parse=False).load_data()

    set_doc_service(DocService(docs)) # 'cache' the documents for later use by our ContextEnricher

    splitter_registry = CodeSplitterRegistry(chunk_lines=100, chunk_lines_overlap=25, max_chars=3000)

    pipeline = IngestionPipeline(
        transformations=[
            IdentitySplitter(),
            SummaryExtractor(llm=language_model or Settings.llm, num_workers=2),
            CodeSplitter(splitter_registry=splitter_registry), 
            SolutionExtractor(), 
            SummaryExtractor(llm=language_model or Settings.llm, prompt_template=SUMMARY_EXTRACT_TEMPLATE, num_workers=2),
            EntityExtractor(llm=language_model or Settings.llm, num_workers=2),
            QuestionsAnsweredExtractor(llm=language_model or Settings.llm, questions=3, num_workers=1),
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

    return nodes
