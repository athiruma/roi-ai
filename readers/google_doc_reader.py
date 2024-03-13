from llama_index.readers.google import GoogleDocsReader


class GoogleDocReader:

    def __init__(self):
        self.__loader = GoogleDocsReader()
        self.__doc_ids = []

    def set_doc_ids(self, doc_ids: list):
        self.__doc_ids = doc_ids

    def get_documents(self):
        return self.__loader.load_data(document_ids=self.__doc_ids)
