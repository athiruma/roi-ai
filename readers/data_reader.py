import os
from ast import literal_eval

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, \
    service_context

from Database.pgvector import PgVector
from readers.google_doc_reader import GoogleDocReader


class DataReader:

    def __init__(self):
        self.__index = None

    def get_index(self, reader: str = ''):
        if reader == 'google_doc_reader':
            data_reader = GoogleDocReader()
            data_reader.set_doc_ids(literal_eval(os.environ['DOC_IDS']))
            index = VectorStoreIndex.from_documents(data_reader.get_documents())
        elif reader == 'vector_db':

            index = VectorStoreIndex.from_vector_store(vector_store=PgVector.get_vector_context())
        elif reader == 'pdf_reader':
            documents = SimpleDirectoryReader(input_files=[os.environ['PDF_PATH']]).load_data()
            index = VectorStoreIndex.from_documents(documents=documents)
        else:
            documents = SimpleDirectoryReader("./sample_data").load_data()
            index = VectorStoreIndex.from_documents(documents=documents)
        self.__index = index
        return self.__index

    def dump_data_to_vector_db(self):
        documents = SimpleDirectoryReader("./sample_data").load_data()

        vector_store = PgVector.get_vector_context()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )
        # storage_context = StorageContext.from_defaults(persist_dir="vector-db")
        # return load_index_from_storage(storage_context=storage_context)
        return index

    def persist_as_vector_data(self):
        self.__index.storage_context.persist(persist_dir="vector-db")
