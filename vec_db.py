from typing import Dict, List, Annotated
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import csv


DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def _load_index(self):
        # Temporary variables to hold the sections
        current_section = None
        current_level1_idx = None
        current_level2_idx = None

        # Open the file and read it
        with open(self.index_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                # Handling different sections based on headers
                if row[0] == "Level 1 Centroids":
                    current_section = "level1_centroids"
                    continue
                elif row[0] == "Nested Centroids (Level 2)":
                    current_section = "nested_centroids"
                    continue
                elif row[0] == "Inverted Lists (Level 1)":
                    current_section = "inverted_lists_level1"
                    continue
                elif row[0] == "Inverted Lists (Level 2)":
                    current_section = "inverted_lists_level2"
                    continue

                # Handling data based on the current section
                if current_section == "level1_centroids":
                    # Read centroids for Level 1
                    self.centroids_level1.append([float(x) for x in row])
                elif current_section == "nested_centroids":
                    # Read nested centroids (Level 2)
                    if 'Cluster' in row[0]:
                        current_level1_idx = int(row[0].split()[-1])
                        self.nested_centroids[current_level1_idx] = []
                    else:
                        self.nested_centroids[current_level1_idx].append([float(x) for x in row])
                elif current_section == "inverted_lists_level1":
                    # Read inverted lists for Level 1
                    cluster_idx = int(row[0])
                    vector_ids = list(map(int, row[1:]))
                    self.inverted_lists_level1[cluster_idx] = vector_ids
                elif current_section == "inverted_lists_level2":
                    # Read inverted lists for Level 2
                    if 'Cluster' in row[0]:
                        current_level1_idx = int(row[0].split()[-1])
                        self.nested_inverted_lists[current_level1_idx] = {}
                    else:
                        cluster_idx = int(row[0])
                        vector_ids = list(map(int, row[1:]))
                        self.nested_inverted_lists[current_level1_idx][cluster_idx] = vector_ids
        self._save_index()


    def __init__(self, database_file_path = "saved_db.csv", index_file_path = "index.csv", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.nlist = 20
        self.centroids_level1 = []
        self.inverted_lists_level1 = {}
        self.nested_centroids = {}
        self.nested_inverted_lists = {}

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self._load_index()
    
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
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        # Compute Level 1 centroid scores and retrieve top-n
        centroid_scores_level1 = np.dot(self.centroids_level1, query.T).flatten()
        closest_level1_centroids = np.argsort(-centroid_scores_level1)[:2]  # Select top 2

        # Aggregate candidate vectors from Level 2 clusters
        candidate_vectors = []
        for level1_idx in closest_level1_centroids:
            if level1_idx in self.nested_centroids:
                level2_centroids = self.nested_centroids[level1_idx]
                centroid_scores_level2 = np.dot(level2_centroids, query.T).flatten()
                closest_level2_centroids = np.argsort(-centroid_scores_level2)[:top_k]

                # Collect vector IDs from Level 2 inverted lists
                for centroid_idx in closest_level2_centroids:
                    candidate_vectors.extend(
                        self.nested_inverted_lists[level1_idx].get(centroid_idx, [])
                    )
        
        # Compute scores for final candidates
        candidate_vectors = list(set(candidate_vectors))  # Remove duplicates
        all_vectors = self.get_all_rows()[candidate_vectors]
        scores = np.dot(all_vectors, query.T).flatten()
        top_candidates = np.argsort(-scores)[:top_k]
        
        return [(scores[i], candidate_vectors[i]) for i in top_candidates]

    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    

    #############################################################################

    def _save_index(self):
        # Saving the Level 1 centroids (top-level centroids)
        with open(self.index_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Save Level 1 centroids
            writer.writerow(["Level 1 Centroids"])
            for centroid in self.centroids_level1:
                writer.writerow(centroid)
            
            writer.writerow([])  # Blank line to separate sections (optional)
            
            # Save the nested centroids (Level 2 sub-clusters for each Level 1 cluster)
            writer.writerow(["Nested Centroids (Level 2)"])
            for level1_idx, level2_centroids in self.nested_centroids.items():
                writer.writerow([f"Level 1 Cluster {level1_idx}"])
                for centroid in level2_centroids:
                    writer.writerow(centroid)
            
            writer.writerow([])  # Blank line to separate sections (optional)

            # Save the inverted lists for Level 1 clusters
            writer.writerow(["Inverted Lists (Level 1)"])
            for cluster_idx, vector_ids in self.inverted_lists_level1.items():
                writer.writerow([cluster_idx] + vector_ids)
            
            writer.writerow([])  # Blank line to separate sections (optional)

            # Save the inverted lists for Level 2 sub-clusters
            writer.writerow(["Inverted Lists (Level 2)"])
            for level1_idx, inverted_lists_level2 in self.nested_inverted_lists.items():
                writer.writerow([f"Level 1 Cluster {level1_idx}"])
                for cluster_idx, vector_ids in inverted_lists_level2.items():
                    writer.writerow([cluster_idx] + vector_ids)

    def _build_index(self):

        num_records = self._get_num_records()
        if num_records == 0:
            raise ValueError("The database is empty. Add records before building the index.")
        
        data = self.get_all_rows()

        # Level 1 clustering (coarse clusters)
        minibatch_kmeans_level1 = MiniBatchKMeans(n_clusters=self.nlist, random_state=DB_SEED_NUMBER, batch_size=1000)
        minibatch_kmeans_level1.fit(data)
        self.centroids_level1 = minibatch_kmeans_level1.cluster_centers_
        
        # Create inverted lists for Level 1
        cluster_assignments_level1 = minibatch_kmeans_level1.labels_
        self.inverted_lists_level1 = {i: [] for i in range(self.nlist)}
        for idx, cluster_idx in enumerate(cluster_assignments_level1):
            self.inverted_lists_level1[cluster_idx].append(idx)
        
        # Now, for each cluster in Level 1, perform Level 2 clustering
        self.nested_centroids = {}
        self.nested_inverted_lists = {}
        
        for level1_idx, level1_vectors in self.inverted_lists_level1.items():
            # Get the vectors assigned to this cluster
            level1_cluster_vectors = data[level1_vectors]
            
            # Level 2 clustering (sub-clusters within each Level 1 cluster)
            minibatch_kmeans_level2 = MiniBatchKMeans(n_clusters=self.nlist // 2, random_state=DB_SEED_NUMBER, batch_size=1000)
            minibatch_kmeans_level2.fit(level1_cluster_vectors)
            
            level2_centroids = minibatch_kmeans_level2.cluster_centers_
            self.nested_centroids[level1_idx] = level2_centroids
            
            # Create inverted lists for Level 2
            cluster_assignments_level2 = minibatch_kmeans_level2.labels_
            inverted_lists_level2 = {i: [] for i in range(self.nlist // 2)}
            for idx, cluster_idx in enumerate(cluster_assignments_level2):
                inverted_lists_level2[cluster_idx].append(level1_vectors[idx])  # Store original level 1 indices
            
            self.nested_inverted_lists[level1_idx] = inverted_lists_level2
        
        # Save the nested structure (if needed)
        self._save_index()

