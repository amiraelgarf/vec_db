from typing import Dict, List, Annotated
import numpy as np
#from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import hnswlib  
import os
from joblib import Parallel, delayed

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
        vectors = self._normalize_vectors(vectors)  # Precompute normalized vectors
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        rows = self._normalize_vectors(rows)
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

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
    
    def retrieve(self, query: np.ndarray, top_k=5) -> List[int]:
        query = self._normalize_vectors(np.array([query]))[0]
        cluster_ids = np.argsort(np.linalg.norm(self.cluster_manager.centroids - query, axis=1))[:max(3, top_k // 2)]
        
        results = []
        for cluster_id in cluster_ids:
            if cluster_id in self.hnsw_indices:
                cluster_results = self.hnsw_indices[cluster_id].query(query, k=top_k)
                results.extend([(self._cal_score(query, self.get_one_row(idx)), idx) for idx in cluster_results])
        
        results.sort(reverse=True)
        return [idx for _, idx in results[:top_k]]

    
    def _cal_score(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2)  # Cosine similarity as vectors are normalized


    def _build_index(self) -> None:
        vectors = self.get_all_rows()
        self.cluster_manager = ClusterManager(
            num_clusters=max(1, min(len(vectors), int(np.sqrt(len(vectors) / 2)))),
            dimension=DIMENSION
        )
        self.cluster_manager.cluster_vectors(vectors)

        cluster_vectors = {i: [] for i in range(self.cluster_manager.num_clusters)}
        for idx, cluster_id in enumerate(self.cluster_manager.assignments):
            cluster_vectors[cluster_id].append(vectors[idx])

        # Balance clusters
        cluster_vectors = self._balance_clusters(cluster_vectors)

        # Build HNSW indices
        self.hnsw_indices = {}
        for cluster_id, cluster_vecs in cluster_vectors.items():
            if len(cluster_vecs) > 0:
                hnsw_index = HNSWIndex(DIMENSION)
                hnsw_index.build(np.array(cluster_vecs))
                self.hnsw_indices[cluster_id] = hnsw_index

    def _balance_clusters(self, cluster_vectors):
        threshold = 10  
        balanced_vectors = {}
        small_clusters = []

        for cluster_id, vecs in cluster_vectors.items():
            if len(vecs) < threshold:
                small_clusters.extend(vecs)
            else:
                balanced_vectors[cluster_id] = vecs

        # Assign small clusters to the nearest large cluster
        if small_clusters:
            for vec in small_clusters:
                # Find the nearest cluster (simple dot-product similarity for demonstration)
                closest_cluster = max(
                    balanced_vectors.keys(),
                    key=lambda cid: np.mean([np.dot(vec, cvec) for cvec in balanced_vectors[cid]])
                )
                balanced_vectors[closest_cluster].append(vec)

        return balanced_vectors


# class ClusterManager:
#     def __init__(self, num_clusters: int, dimension: int):
#         self.num_clusters = num_clusters
#         self.dimension = dimension
#         self.kmeans = None
#         self.centroids = None
#         self.assignments = None

#     def cluster_vectors(self, vectors: np.ndarray) -> None:
#         self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=DB_SEED_NUMBER, batch_size=1024)
#         self.assignments = self.kmeans.fit_predict(vectors)
#         self.centroids = self.kmeans.cluster_centers_

class ClusterManager:
    def __init__(self, num_clusters, dimension):
        self.num_clusters = num_clusters
        self.dimension = dimension
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)  # Set n_init explicitly

    def cluster_vectors(self, vectors):
        self.assignments = self.kmeans.fit_predict(vectors)
class HNSWIndex:
    def __init__(self, dimension: int, max_elements=1000, ef_construction=50, m=16):
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=m)

    def build(self, vectors: np.ndarray) -> None:
        self.index.add_items(vectors)

    def query(self, vector: np.ndarray, k: int) -> List[int]:
        labels, _ = self.index.knn_query(vector, k=k)
        return labels[0]
