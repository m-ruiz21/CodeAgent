from typing import List
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.schema import TextNode

class IdentitySplitter(TextSplitter):
    """
    A splitter that directly converts a document into a single TextNode without splitting.
    """

    def split_text(self, text: str) -> List[str]:
        """
        Converts the input text into a single TextNode.

        :param text: The input text to be converted.
        :return: A list containing one TextNode.
        """
        return [text]