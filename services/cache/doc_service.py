class DocService:
    def __init__(self, docs):
        self._docs = {}
        for doc in docs:
            file_path = doc.metadata.get('file_path')
            content = getattr(doc, 'text', None) or getattr(doc, 'content', None)

            if not (file_path and content):
                raise ValueError(f"Document must have 'file_path' and 'text' or 'content': {doc}")

            self._docs[file_path] = content

    def get_content(self, file_path):
        return self._docs.get(file_path)

    def set_content(self, file_path, content):
        self._docs[file_path] = content

# Singleton instance management
_doc_service_instance: DocService | None = None

def set_doc_service(instance: DocService):
    global _doc_service_instance
    _doc_service_instance = instance

def get_doc_service() -> DocService:
    return _doc_service_instance