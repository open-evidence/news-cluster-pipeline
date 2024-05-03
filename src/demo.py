from ClusteringPipeline import ClusteringPipeline
import pandas as pd


df = pd.read_csv('src/news1000.csv')
df.to_csv('src/news1000.csv', index=False)
print(len(df))

hdb_config = {
    'min_cluster_size': 2,
    'min_samples': 1,
    'cluster_selection_epsilon': 0.2,
    'cluster_selection_method': 'leaf',
    'metric': 'euclidean',
}

pipeline = ClusteringPipeline(hdb_config, model_name='BAAI/bge-m3', title_col='title', datetime_col='pubdate')
pipeline.run(df)


pipeline.print_clusters()
print('unclustered:', len(pipeline.labels[pipeline.labels == -1]))