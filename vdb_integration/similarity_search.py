import time
from pinecone import Pinecone, ServerlessSpec


class AudioSimilaritySearch:
    def __init__(self, api_key, index_name="model-openl3"):
        self.pc = Pinecone(api_key=api_key)
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=512,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-1")
            )
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        self.index = self.pc.Index(index_name)

    def search(self, vector, top_k=1, namespace="ns1"):
        return self.index.query(
            vector=vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        ) 