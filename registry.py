from llama_index.core.node_parser import CodeSplitter

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
            "cs": "c_sharp",
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
        }
    
    
    def get_splitter(self, path: str) -> CodeSplitter:
        file_extension = path.split('.')[-1]
        if file_extension not in self._language_map:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        language = self._language_map[file_extension]
        if language in self.codeSplitters:
            return self.codeSplitters[language]

        self.codeSplitters[language] = CodeSplitter.from_defaults(
            language=language,
            **self.splitter_params
        )

        return self.codeSplitters[language]