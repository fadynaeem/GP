import time
from pinecone import Pinecone, ServerlessSpec
from Config import Config


class VectorDBService:
    def __init__(self):
        self.api_key = Config.PINECONE_API_KEY
        self.index_name = Config.PINECONE_INDEX_NAME
        self.dimensions = Config.VECTOR_DIMENSION
        self.cloud = Config.PINECONE_CLOUD
        self.region = Config.PINECONE_REGION
        self.pc = None
        self.index = None
        self.initialize_connection()
        
    def initialize_connection(self):
        self.pc = Pinecone(api_key=self.api_key)
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            spec = ServerlessSpec(cloud=self.cloud, region=self.region)
            self.pc.create_index(
                name=self.index_name, 
                dimension=self.dimensions, 
                metric="cosine", 
                spec=spec
            )
            self._wait_for_index_readiness()
        else:
            print(
                f"Index '{self.index_name}' already exists. "
                "Connecting to it."
            )
        
        self.index = self.pc.Index(self.index_name)

    def _wait_for_index_readiness(self):
        while True:
            description = self.pc.describe_index(self.index_name)
            if description["status"]["ready"]:
                break
            time.sleep(1)
    
    def describe_index_stats(self):
        return self.index.describe_index_stats()
    
    def query_nearest_neighbors(self, vector, k=1, namespace="ns1"):
        query_result = self.index.query(
            vector=vector,
            top_k=k,
            include_values=False,
            include_metadata=True,
            namespace=namespace
        )
        return query_result
        
    def filter_results(self, result):
        filtered_results = []
        if "matches" in result:
            for match in result["matches"]:
                metadata = match.get("metadata", {})
                filtered_metadata = {
                    "AyahNo": metadata.get("AyahNo"),
                    "SurahNo": metadata.get("SurahNo")
                }
                filtered_results.append(filtered_metadata)
        return filtered_results