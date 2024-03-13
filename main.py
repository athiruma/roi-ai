import json
import os
import textwrap

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import TimeWeightedPostprocessor
from llama_index.core.settings import Settings
from llama_index.embeddings.instructor import InstructorEmbedding

from config import GOOGLE_DOC_READER
from readers.data_reader import DataReader
from roi_ai import RioAI


Settings.llm = RioAI()
data_reader = DataReader()



embed_model = InstructorEmbedding(model_name="hkunlp/instructor-xl")

Settings.embed_model = embed_model
# Define embed model (if needed, comment out if using a remote LLM)
# Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# Load your data (if using SummaryIndex)
#



# parser = SimpleNodeParser()
# nodes = parser.get_nodes_from_documents(documents)

# os.environ['DOC_IDS'] = '["1hegHsgkzNosl-HhD0fCBYztzpI3h-TbHBJLLyaQqjUM"]'
# index = data_reader.get_index(reader=GOOGLE_DOC_READER)

index = data_reader.dump_data_to_vector_db()

# Query and print response (if using SummaryIndex)
# query_engine = index.as_query_engine()
query_engine = index.as_query_engine()
response = query_engine.query("Summary of the document")

#rajesh = index.as_chat_engine()
#response = query_engine.query("write me the summary about sachin in more detail")
print(textwrap.fill(str(response), 100))

