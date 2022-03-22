# GHRS: Graph-based hybrid recommendation system

Partial implementation for : [GHRS: Graph-based hybrid recommendation system with application to movie recommendation](https://doi.org/10.1016/j.eswa.2022.116850) [Updated pdf on [arXiv](https://doi.org/10.48550/arXiv.2111.11293)]

**GHRS** is a Graph-based hybrid recommendation system for movie recommendation
> Research about recommender systems emerges over the last decade and comprises valuable services to increase different companies' revenue. Several approaches exist in handling paper recommender systems. While most existing recommender systems rely either on a content-based approach or a collaborative approach, there are hybrid approaches that can improve recommendation accuracy using a combination of both approaches. Even though many algorithms are proposed using such methods, it is still necessary for further improvement. In this paper, we propose a recommender system method using a graph-based model associated with the similarity of users' ratings, in combination with users' demographic and location information. By utilizing the advantages of Autoencoder feature extraction, we extract new features based on all combined attributes. Using the new set of features for clustering users, our proposed approach (GHRS) has gained a significant improvement, which dominates other methods' performance in the cold-start problem. The experimental results on the MovieLens dataset show that the proposed algorithm outperforms many existing recommendation algorithms on recommendation accuracy. [1]



# Scripts

Feature100K.py: 
creates similarity graph between users, extracts graph features and generates the final feature vector by combining the graph features and categorized side information for users on dataset MovieLens 100K [2].

Feature1M.py: 
creates similarity graph between users, extracts graph features and generates the final feature vector by combining the graph features and categorized side information for users on dataset MovieLens 1M [3].

# Data

     |-datasets
     | |-ml-100k	# MovieLens 100K dataset files
     | |-ml-1m		# MovieLens 1M dataset files
     |-data100k		# Combined features (graph features and side information) for specific value of alpha for dataset MovieLens 100K
     | |-x_train_alpha(0.005).pkl
     | |-x_train_alpha(0.01).pkl
     | |...
     |-data1m		# Combined features (graph features and side information) for specific value of alpha for dataset MovieLens 1M
     | |-x_train_alpha(0.005).pkl
     | |-x_train_alpha(0.01).pkl
     | |...
     | ...

