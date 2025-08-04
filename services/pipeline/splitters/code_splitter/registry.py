from llama_index.core.node_parser import CodeSplitter, JSONNodeParser, MarkdownNodeParser, HTMLNodeParser, TokenTextSplitter 
from llama_index.core.node_parser.interface import NodeParser

DEFAULT_CHUNK_LINES = 40
DEFAULT_LINES_OVERLAP = 15
DEFAULT_MAX_CHARS = 1500

class CodeSplitterRegistry():  
    def __init__(self, chunk_lines = DEFAULT_CHUNK_LINES, chunk_lines_overlap = DEFAULT_LINES_OVERLAP, max_chars = DEFAULT_MAX_CHARS) -> None:
        self.splitter_params = {
            "chunk_lines": chunk_lines,
            "chunk_lines_overlap": chunk_lines_overlap,
            "max_chars": max_chars
        } 
        self.codeSplitters = {} 
        self._language_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "jsx": "javascript",
            "tsx": "typescript",
            "rb": "ruby",
            "rs": "rust",
            "go": "go",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "cc": "cpp",
            "h": "c",
            "hpp": "cpp",
            "cs": "csharp",
            "php": "php",
            "scala": "scala",
            "swift": "swift",
            "kt": "kotlin",
            "lua": "lua",
            "hs": "haskell",
            "ml": "ocaml",
            "sh": "bash",
            "yaml": "yaml",
            "yml": "yaml",
            "json": "json",
            "md": "markdown",
            "html": "html",
            "css": "css",
            "scss": "scss",
            "sass": "scss",
            "sql": "sql",
            "proto": "proto",
            "elm": "elm",
            "clj": "clojure",
            "ex": "elixir",
            "exs": "elixir",
            "txt": "text",  # generic text files
        }
    
    def get_splitter(self, path: str) -> CodeSplitter | NodeParser:
        """Get a CodeSplitter for the given file path, creating it if necessary."""
        file_extension = path.split('.')[-1]
        if not self.is_supported(path):
            raise ValueError(f"Unsupported file extension: {file_extension}")

        language = self._language_map[file_extension]
        
        if language in self.codeSplitters:
            return self.codeSplitters[language]

        if language == "json":
            self.codeSplitters[language] = JSONNodeParser().from_defaults()

        elif language == "markdown":
            self.codeSplitters[language] = MarkdownNodeParser().from_defaults()

        elif language == "html":
            self.codeSplitters[language] = HTMLNodeParser().from_defaults()
        elif language == "text":
            self.codeSplitters[language] = TokenTextSplitter(    
                chunk_size=4000,                  
                chunk_overlap=200,
                backup_separators=["\n\n", "\n", " "],
            )

        else:
            self.codeSplitters[language] = CodeSplitter.from_defaults(
                language=language,
                **self.splitter_params
            )

        return self.codeSplitters[language]
    
    def is_supported(self, path: str) -> bool:
        """Check if a file extension is supported by the registry."""
        file_extension = path.split('.')[-1]
        return file_extension in self._language_map
