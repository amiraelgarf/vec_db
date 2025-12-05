import numpy as np
import os
import struct
from sklearn.cluster import MiniBatchKMeans
from typing import Annotated
import gc

# Constants
DIMENSION = 64
ELEMENT_SIZE = 4  # float32 is 4 bytes
DB_SEED_NUMBER = 42

class VecDB:
    # -------------------------------------------------------------------------
    # 1. STRICT INIT SIGNATURE (Do not change this)
    # -------------------------------------------------------------------------
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat",
                 new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")

            # Clean up old files
            if os.path.exists(self.db_path): os.remove(self.db_path)
            if os.path.exists(self.index_path): os.remove(self.index_path)

            self.generate_database(db_size)
        else:
            # If we are loading an existing DB but the index is missing, build it.
            if os.path.exists(self.db_path) and not os.path.exists(self.index_path):
                print("[INIT] Database found but Index missing. Building Index...")
                self._build_index()

    # -------------------------------------------------------------------------
    # 2. FILE OPERATIONS
    # -------------------------------------------------------------------------
    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r',
                                    shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return np.zeros(DIMENSION)

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def _get_num_records(self) -> int:
        if not os.path.exists(self.db_path): return 0
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    # -------------------------------------------------------------------------
    # 3. GENERATION & INDEXING
    # -------------------------------------------------------------------------
    def generate_database(self, size: int) -> None:
        print(f"[DB] Generating {size} vectors...")
        rng = np.random.default_rng(DB_SEED_NUMBER)

        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=(size, DIMENSION))

        chunk_size = 500_000
        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            mmap_vectors[i:end] = rng.random((end - i, DIMENSION), dtype=np.float32)
            if i % 1_000_000 == 0: mmap_vectors.flush()

        mmap_vectors.flush()
        print("[DB] Generation complete.")
        self._build_index()

    def _build_index(self):
        num_records = self._get_num_records()
        print(f"[INDEX] Building Single-File Index for {num_records} vectors...")

        # A. Determine clusters
        if num_records <= 1_000_000: n_clusters = 1000
        elif num_records <= 10_000_000: n_clusters = 3000
        else: n_clusters = 4000

        # B. Train K-Means (Subsampling)
        print("[INDEX] Training K-Means...")
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000,
                                 random_state=DB_SEED_NUMBER, n_init='auto')

        train_size = min(500_000, num_records)
        kmeans.fit(mmap_vectors[:train_size])

        centroids = kmeans.cluster_centers_.astype(np.float32)

        # C. Assign Vectors
        print("[INDEX] Assigning vectors...")
        batch_size = 100000
        all_labels = np.zeros(num_records, dtype=np.int32)

        for i in range(0, num_records, batch_size):
            end = min(i + batch_size, num_records)
            all_labels[i:end] = kmeans.predict(mmap_vectors[i:end])

        # D. Sort IDs
        print("[INDEX] Sorting lists...")
        sorted_indices = np.argsort(all_labels)
        sorted_labels = all_labels[sorted_indices]

        # E. WRITE SINGLE INDEX FILE
        # Format:
        # [N_Clusters (int)]
        # [Centroids (N*Dim floats)]
        # [Offset_Table (N*2 ints -> start_byte, count)]
        # [Inverted Lists (Integers...)]

        print(f"[INDEX] Writing to {self.index_path}...")
        with open(self.index_path, "wb") as f:
            # 1. Write Header: Number of Clusters
            f.write(struct.pack("I", n_clusters))

            # 2. Write Centroids
            f.write(centroids.tobytes())

            # 3. Reserve space for Offset Table
            # Each entry is 2 ints (start_offset, count) -> 8 bytes
            table_offset_start = f.tell()
            f.write(b'\0' * (n_clusters * 8))

            # 4. Write Inverted Lists & Record Offsets
            cluster_metadata = [] # Stores (offset, count)

            for cid in range(n_clusters):
                # Find range in sorted array
                start_idx = np.searchsorted(sorted_labels, cid, side='left')
                end_idx = np.searchsorted(sorted_labels, cid, side='right')

                count = end_idx - start_idx
                current_file_pos = f.tell()

                # Store metadata (Where this list starts, How many items)
                cluster_metadata.append((current_file_pos, count))

                if count > 0:
                    ids = sorted_indices[start_idx:end_idx].astype(np.int32)
                    f.write(ids.tobytes())

            # 5. Go back and fill in the Offset Table
            f.seek(table_offset_start)
            for offset, count in cluster_metadata:
                f.write(struct.pack("II", offset, count))

        print("[INDEX] Done.")

    # -------------------------------------------------------------------------
    # 4. RETRIEVAL
    # -------------------------------------------------------------------------
    def retrieve(self, query: np.ndarray, top_k=5):
        query = query.reshape(-1).astype(np.float32)
        q_norm = np.linalg.norm(query)
        
        num_records = self._get_num_records()
        if num_records <= 1_000_000: n_probes = 5
        else: n_probes = 5

        # --- A & B. Coarse Search (Metadata + Centroids) ---
        # The reading of the index file (centroids and offsets) is fast 
        # and necessary, and kept outside the main loop.
        with open(self.index_path, "rb") as f:
            n_clusters = struct.unpack("I", f.read(4))[0]
            centroid_bytes = f.read(n_clusters * DIMENSION * ELEMENT_SIZE)
            centroids = np.frombuffer(centroid_bytes, dtype=np.float32).reshape(n_clusters, DIMENSION)
            table_bytes = f.read(n_clusters * 8)
            cluster_table = np.frombuffer(table_bytes, dtype=np.uint32).reshape(n_clusters, 2)
            
            c_norms = np.linalg.norm(centroids, axis=1)
            dists = np.dot(centroids, query)
            sims = dists / (c_norms * q_norm + 1e-10)
            closest_clusters = np.argsort(sims)[::-1][:n_probes]
            
            del centroids, centroid_bytes, c_norms, dists, sims

            import heapq
            # --- C. Fine Search (Optimized Batches) ---
            top_heap = [] 
            
            # 1. ACCELERATION STEP: Open the DB file ONCE
            # This eliminates the overhead of opening/closing the file per batch.
            with open(self.db_path, "rb") as db_file:
                
                # Use a larger batch size (e.g., 50k or 100k) to reduce I/O calls
                # 100k vectors * 128D * 4 bytes/float ~ 50 MB RAM per batch.
                # This is a controlled increase in RAM usage.
                batch_size = 50000 
                
                for cid in closest_clusters:
                    offset, count = cluster_table[cid]
                    if count == 0: continue

                    # Read vector IDs for this cluster (from index_path file 'f')
                    f.seek(int(offset))
                    ids_bytes = f.read(int(count) * 4)
                    row_ids = np.frombuffer(ids_bytes, dtype=np.int32)

                    # Process vectors in optimized batches
                    for batch_start in range(0, len(row_ids), batch_size):
                        batch_end = min(batch_start + batch_size, len(row_ids))
                        batch_ids = row_ids[batch_start:batch_end]
                        
                        # Compute the byte offset for the FIRST vector in this batch
                        # Assumes your database file is a flat array of vectors [v0, v1, v2, ...]
                        # first_vid = batch_ids[0]
                        # first_vector_byte_offset = int(first_vid) * DIMENSION * ELEMENT_SIZE
                        
                        # # Compute the total bytes to read for all vectors in this batch
                        # num_vectors_to_read = len(batch_ids)
                        # total_bytes_to_read = num_vectors_to_read * DIMENSION * ELEMENT_SIZE
                        
                        # # --- DRAMATIC PERFORMANCE BOOST ---
                        # # 2. Seek to the start of the FIRST vector in the batch
                        # db_file.seek(first_vector_byte_offset, os.SEEK_SET)
                        
                        # # 3. Read the ENTIRE contiguous block of vectors for the batch
                        # vec_bytes = db_file.read(total_bytes_to_read)
                        
                        # # 4. Load the whole batch into RAM at once
                        # batch_vecs = np.frombuffer(vec_bytes, dtype=np.float32).reshape(
                        #     num_vectors_to_read, DIMENSION
                        # )
                        # --- Build contiguous runs ---
                        runs = []
                        run_start = int(batch_ids[0])
                        prev = run_start
                        for vid in batch_ids[1:]:
                            vid = int(vid)
                            if vid == prev + 1:
                                prev = vid
                                continue
                            runs.append((run_start, prev))
                            run_start = vid
                            prev = vid
                        runs.append((run_start, prev))

                        # Allocate buffer
                        batch_vecs = np.empty((len(batch_ids), DIMENSION), dtype=np.float32)

                        write_pos = 0
                        for (s_vid, e_vid) in runs:
                            run_len = e_vid - s_vid + 1
                            file_offset = int(s_vid) * DIMENSION * ELEMENT_SIZE

                            db_file.seek(file_offset)
                            read_bytes = db_file.read(run_len * DIMENSION * ELEMENT_SIZE)

                            run_vectors = np.frombuffer(read_bytes, dtype=np.float32).reshape(run_len, DIMENSION)
                            batch_vecs[write_pos:write_pos + run_len] = run_vectors
                            write_pos += run_len

                        # Compute scores for this batch (FAST: RAM-based vectorized math)
                        vec_norms = np.linalg.norm(batch_vecs, axis=1)
                        dot_products = np.dot(batch_vecs, query)
                        batch_scores = dot_products / (vec_norms * q_norm + 1e-10)
                        
                        # Update top-k heap (Heap is very fast)
                        for idx, score in enumerate(batch_scores):
                            vid = int(batch_ids[idx])
                            if len(top_heap) < top_k:
                                heapq.heappush(top_heap, (score, vid))
                            elif score > top_heap[0][0]:
                                heapq.heapreplace(top_heap, (score, vid))
                        
                        # Free batch memory
                        del batch_vecs, vec_norms, dot_products, batch_scores

            # --- D. Extract Final Top K ---
            if not top_heap: return []
            
            top_heap.sort(key=lambda x: x[0], reverse=True)
            return [vid for score, vid in top_heap]