import time
from pinecone import Pinecone, ServerlessSpec


class PineConeVdb:
    def __init__(
        self, api_key, index_name="verse-index", dimensions=768,
        cloud="aws", region="us-west-2"
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimensions = dimensions
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        self.index = self.pc.Index(index_name)

    def describe_index_stats(self):
        return self.index.describe_index_stats()

    def get_knn(self, k, vector, namespace="ns1"):
        return self.index.query(
            vector=vector,
            top_k=k,
            include_values=False,
            include_metadata=True,
            namespace=namespace
        ) 