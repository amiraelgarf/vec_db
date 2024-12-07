from typing import Dict, List, Annotated
import numpy as np
from sklearn.cluster import KMeans
import hnswlib  
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: np.ndarray, top_k=5):
        # Step 1: Identify nearest clusters
        cluster_ids = np.argsort(
            np.linalg.norm(self.cluster_manager.centroids - query, axis=1)
        )[:5]  # Top 5 clusters
    
        # Step 2: Retrieve candidates from clusters
        results = []
        for cluster_id in cluster_ids:
            cluster_results = self.hnsw_indices[cluster_id].query(query, k=top_k)
            results.extend([(self._cal_score(query, self.get_one_row(idx)), idx) for idx in cluster_results])
    
        # Step 3: Rank and return top-k
        results.sort(reverse=True)
        return [idx for _, idx in results[:top_k]]

    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Load all vectors
        vectors = self.get_all_rows()
        self.cluster_manager = ClusterManager(num_clusters=10000, dimension=DIMENSION)
        self.cluster_manager.cluster_vectors(vectors)
        # Create an HNSW graph for each cluster
        self.hnsw_indices = {i: HNSWIndex(DIMENSION) for i in range(self.cluster_manager.num_clusters)}
        for cluster_id in range(self.cluster_manager.num_clusters):
            cluster_vectors = vectors[self.cluster_manager.assignments == cluster_id]
            self.hnsw_indices[cluster_id].build(cluster_vectors)

class ClusterManager:
    def __init__(self, num_clusters: int, dimension: int):
        self.num_clusters = None  # This will be set later based on the data size
        self.dimension = dimension
        self.kmeans = None
        self.centroids = None
        self.assignments = None

    def cluster_vectors(self, vectors: np.ndarray) -> None:
        # Adjust the number of clusters to be at most the number of vectors
        self.num_clusters = min(len(vectors), 10000)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=DB_SEED_NUMBER, n_init=10)  
        self.assignments = self.kmeans.fit_predict(vectors)
        self.centroids = self.kmeans.cluster_centers_

    def assign_to_cluster(self, vector: np.ndarray) -> int:
        return np.argmin(np.linalg.norm(self.centroids - vector, axis=1))

class HNSWIndex:
    def __init__(self, dimension: int, max_elements=1000, ef_construction=100, m=8):
        self.dimension = dimension
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=m)
    
    def build(self, vectors: np.ndarray):
        self.index.add_items(vectors)
    
    def query(self, vector: np.ndarray, k: int) -> List[int]:
        labels, distances = self.index.knn_query(vector, k=k)
        return labels[0]
