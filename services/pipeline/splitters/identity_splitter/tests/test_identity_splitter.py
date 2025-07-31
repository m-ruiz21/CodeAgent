import unittest
from services.pipeline.identity_splitter.identity_splitter import IdentitySplitter
from llama_index.core.schema import TextNode

class TestIdentitySplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = IdentitySplitter()

    def test_split_text(self):
        text = "This is a test document."
        result = self.splitter.split_text(text)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextNode)
        self.assertEqual(result[0].text, text)

if __name__ == "__main__":
    unittest.main()
