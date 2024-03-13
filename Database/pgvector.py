from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.legacy.vector_stores import PGVectorStore

from config import *


class PgVector:

    @staticmethod
    def get_vector_context():
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            password=DB_PASSWORD,
            user=DB_USER,
            table_name=DB_TABLE,
            embed_dim=DB_EMBEDDING,  # openai embedding dimension
        )
        return vector_store

    def upload_data(self, row):
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            password=DB_PASSWORD,
            user=DB_USER,
            table_name=DB_TABLE,
            embed_dim=DB_EMBEDDING,  # openai embedding dimension
        )
        return vector_store

