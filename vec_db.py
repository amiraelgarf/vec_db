import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Annotated

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70


class VecDB:
    def __init__(self, database_file_path="saved_db_1m.dat", index_file_path="saved_db_1m", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.cluster_manager = None
        self.pq_codebooks = {}  # For Product Quantization
        self.last_indexed_row = 0 

        # Ensure the index directory exists
        os.makedirs(self.index_path, exist_ok=True)

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self.load_indices()

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index(full_rebuild=True)  # Full rebuild for a new database

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode="w+", shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def load_indices(self) -> None:
        centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
        assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")

        if os.path.exists(centroids_path) and os.path.exists(assignments_path):
            # Load centroids and assignments
            centroids = np.load(centroids_path)
            assignments = np.load(assignments_path)

            self.cluster_manager = ClusterManager(num_clusters=len(centroids), dimension=DIMENSION)
            self.cluster_manager.centroids = centroids
            self.cluster_manager.assignments = assignments
        else:
            raise FileNotFoundError("Centroids or assignments files not found.")

        # Load cluster data for each cluster
        self.pq_codebooks = {}
        for cluster_id in np.unique(self.cluster_manager.assignments):
            cluster_file = os.path.join(self.index_path, f"cluster_{cluster_id}.npz")
            if os.path.exists(cluster_file):
                cluster_data = np.load(cluster_file)
                self.pq_codebooks[cluster_id] = {
                    "ids": cluster_data["ids"],
                    "codes": cluster_data["codes"],
                    "codebook": cluster_data["codebook"]
                }
            else:
                print(f"Warning: Cluster file for cluster {cluster_id} not found.")



    def _build_index(self, full_rebuild=False):
        vectors = self.get_all_rows()

        if full_rebuild:
            self.cluster_manager = ClusterManager(
                num_clusters=max(1, min(len(vectors), int(np.sqrt(len(vectors) / 2)))), dimension=DIMENSION
            )
            self.cluster_manager.cluster_vectors(vectors)

            # Save centroids and assignments to disk
            centroids_path = os.path.join(self.index_path, "ivf_centroids.npy")
            assignments_path = os.path.join(self.index_path, "ivf_assignments.npy")
            np.save(centroids_path, self.cluster_manager.kmeans.cluster_centers_)
            np.save(assignments_path, self.cluster_manager.assignments)

            # Create codebooks and save IDs with PQ codes
            for cluster_id in np.unique(self.cluster_manager.assignments):
                cluster_vector_indices = np.where(self.cluster_manager.assignments == cluster_id)[0]
                cluster_vectors = vectors[cluster_vector_indices]

                # Train PQ codebook
                codebook = self._train_pq_codebook(cluster_vectors)
                pq_codes = np.array([self._quantize(codebook, vec) for vec in cluster_vectors])

                # Save cluster data: IDs, PQ codes, and codebook
                cluster_data = {
                    "ids": cluster_vector_indices,
                    "codes": pq_codes,
                    "codebook": codebook
                }
                cluster_file = os.path.join(self.index_path, f"cluster_{cluster_id}.npz")
                np.savez_compressed(cluster_file, **cluster_data)
        else:
            # Incremental indexing
            new_vectors = vectors[self.last_indexed_row:]
            if len(new_vectors) == 0:
                return  # Nothing to index

            new_assignments = self.cluster_manager.kmeans.predict(new_vectors)

            # Update assignments
            self.cluster_manager.assignments = np.concatenate(
                [self.cluster_manager.assignments, new_assignments]
            )

            # Process each affected cluster
            for cluster_id in np.unique(new_assignments):
                # Combine existing and new vectors for this cluster
                cluster_vector_indices = np.where(self.cluster_manager.assignments == cluster_id)[0]
                cluster_vectors = vectors[cluster_vector_indices]

                # Train PQ codebook
                codebook = self._train_pq_codebook(cluster_vectors)
                pq_codes = np.array([self._quantize(codebook, vec) for vec in cluster_vectors])

                # Save updated cluster data
                cluster_data = {
                    "ids": cluster_vector_indices,
                    "codes": pq_codes,
                    "codebook": codebook
                }
                cluster_file = os.path.join(self.index_path, f"cluster_{cluster_id}.npz")
                np.savez_compressed(cluster_file, **cluster_data)

                # Update PQ codebook in memory
                self.pq_codebooks[cluster_id] = codebook

            self.last_indexed_row = len(vectors)


    def retrieve(self, query: np.ndarray, top_k=5) -> List[int]:
        # Load indices if not already loaded
        if not self.cluster_manager or self.cluster_manager.centroids is None:
            self.load_indices()

        if self.cluster_manager.centroids is None or len(self.cluster_manager.centroids) == 0:
            return []  # No clusters available

        # Calculate cosine similarity between query and centroids
        cluster_scores = [
            (cluster_id, self._cal_score(query, centroid))
            for cluster_id, centroid in enumerate(self.cluster_manager.centroids)
        ]

        # Sort clusters by descending similarity
        sorted_clusters = sorted(cluster_scores, key=lambda x: x[1], reverse=True)
        cluster_ids = [cluster_id for cluster_id, _ in sorted_clusters[:max(3, top_k // 2)]]

        results = []

        # Search within the selected clusters
        for cluster_id in cluster_ids:
            cluster_file = os.path.join(self.index_path, f"cluster_{cluster_id}.npz")
            if not os.path.exists(cluster_file):
                continue  # Skip if cluster data is missing

            # Load cluster data
            cluster_data = np.load(cluster_file)
            cluster_vector_ids = cluster_data["ids"]
            cluster_codes = cluster_data["codes"]
            codebook = cluster_data["codebook"]

            # Perform PQ-based search
            pq_results = self._pq_search(cluster_codes, query, top_k, codebook)
            for local_idx, similarity in pq_results:
                global_id = cluster_vector_ids[local_idx]
                results.append((global_id, similarity))

        # Sort final results by similarity
        results.sort(reverse=True, key=lambda x: x[1])
        return [idx for idx, _ in results[:top_k]]


    def _pq_search(self, codes: np.ndarray, query: np.ndarray, top_k: int, codebook: np.ndarray) -> List[tuple]:
        # Reconstruct vectors using the codebook
        try:
            reconstructed_vectors = np.array([codebook[code] for code in codes])
        except IndexError as e:
            print(f"Error in reconstructing vectors: {e}")
            raise

        # Calculate scores for each vector against the query using `_cal_score`
        scores = [self._cal_score(reconstructed_vec, query) for reconstructed_vec in reconstructed_vectors]

        # Get the top-k indices with the highest similarity scores
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Return the top-k results as tuples of (index, similarity score)
        return [(idx, scores[idx]) for idx in top_indices]





    def _quantize(self, codebook: np.ndarray, vector: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(codebook - vector, axis=1)))


    def _train_pq_codebook(self, cluster_vectors: np.ndarray) -> np.ndarray:
        num_clusters = min(len(cluster_vectors), 256)
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=DB_SEED_NUMBER)
        kmeans.fit(cluster_vectors)
        return kmeans.cluster_centers_


    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
        return np.memmap(self.db_path, dtype=np.float32, mode="r", shape=(num_records, DIMENSION))
    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = int(row_num * DIMENSION * ELEMENT_SIZE)
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve row {row_num}: {e}")
    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity        
class ClusterManager:
    def __init__(self, num_clusters: int, dimension: int):
        self.num_clusters = num_clusters
        self.dimension = dimension
        self.kmeans = None
        self.centroids = None
        self.assignments = None

    def cluster_vectors(self, vectors: np.ndarray) -> None:
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=DB_SEED_NUMBER, batch_size=1024)
        self.assignments = self.kmeans.fit_predict(vectors)
        self.centroids = self.kmeans.cluster_centers_
