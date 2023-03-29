import os
import re
from KoBERTScore import BERTScore
from sklearn.cluster import KMeans
from transformers import BertModel, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser


def get_cluster_kmeans(corpus, num_clusters, model_name):

    embedder = SentenceTransformer(model_name)
    corpus_embeddings = embedder.encode(corpus, batch_size=512, convert_to_numpy=True, show_progress_bar=True)

    # clustering_model = KMeans(n_clusters=num_clusters, random_state=random_state)
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)

    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    # sort
    for i in range(num_clusters):
        clustered_sentences[i] = sorted(clustered_sentences[i])
    """
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i+1)
        print(cluster)
        print("")
    """
    return clustered_sentences


def main():
    parser = ArgumentParser()
    parser.add_argument("--num_clusters", default=10, type=int)
    args = parser.parse_args()

    num_clusters = args.num_clusters
    model_name = "beomi/kcbert-base"

    # read file
    num_tabs = 2
    utterances = []
    data_dir = "./data/"
    input_file = "text_call.tsv"
    with open(os.path.join(data_dir, input_file), "r", encoding="utf-8") as f:
        for l_idx, line in enumerate(f):
            line = line.strip()
            line_split = line.split("\t")
            assert len(line_split) == num_tabs, f"line #{l_idx}: {line} [TAB] ill-formatted"
            utterance = line_split[-1].strip()
            re_matched = re.match(r"SPK(\d+)* ", utterance)
            if re_matched is None:
                print(f"line #{l_idx}: {line} [SPK] ill-formatted")
                continue
            else:
                utterance = utterance[re_matched.span()[-1]:].strip()
                utterances.append(utterance)

    num_clusters = min(num_clusters, len(utterances))
    print(f"num_clusters: {num_clusters}")
    clustered_sentences = get_cluster_kmeans(corpus=utterances, num_clusters=num_clusters, model_name=model_name)

    for c_idx in range(num_clusters):
        print(f"{c_idx} / {num_clusters}")
        with open(os.path.join(data_dir, f"{c_idx}.txt"), "w", encoding="utf-8") as out_f:
            for sent in clustered_sentences[c_idx]:
                out_f.write(f"{sent}\n")


if __name__ == "__main__":
    main()
