import os
import re

import numpy as np
import torch
import multiprocessing

from sklearn.metrics import silhouette_score

from KoBERTScore import BERTScore
from sklearn.cluster import KMeans, MiniBatchKMeans
from transformers import BertModel, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from matplotlib import pyplot as plt
ALLOCATED_GPUS = None


def iterative_cluster(corpus_embeddings, num_cores, min_cluster=10000, max_cluster=100000, interval=10000):

    inertia_list = []
    silhouette_list = []
    for i in range(min_cluster, max_cluster, interval):
        cluster_model = MiniBatchKMeans(n_clusters=i, verbose=1, batch_size=256*num_cores, random_state=0)
        cluster_labels = cluster_model.fit_predict(corpus_embeddings)
        # scores
        inertia_list.append([i, cluster_model.inertia_])
        print(i, cluster_model.inertia_)
        silhouette_avg = silhouette_score(corpus_embeddings, cluster_labels)
        silhouette_list.append([i, silhouette_avg])
        print(i, silhouette_avg)

    return inertia_list, silhouette_list


def visualize_elbow(corpus_embeddings, num_cores, min_cluster=10000, max_cluster=50000, interval=10000):
    sse = []
    for n_cluster in range(min_cluster, max_cluster, interval):
        clustering_model = MiniBatchKMeans(
            n_clusters=n_cluster, verbose=1, batch_size=256 * num_cores, random_state=12345
        )
        clustering_model.fit(corpus_embeddings)
        sse.append([n_cluster, clustering_model.inertia_])
    return sse


def visualize_silhouette_layer(corpus_embeddings, num_cores, min_cluster=10000, max_cluster=50000, interval=10000):

    results = []

    for i in range(min_cluster, max_cluster, interval):
        clustering_model = MiniBatchKMeans(n_clusters=i, verbose=1, batch_size=256 * num_cores)
        cluster_labels = clustering_model.fit_predict(corpus_embeddings)
        silhouette_avg = silhouette_score(corpus_embeddings, cluster_labels)
        results.append([i, silhouette_avg])

    # for print
    best_cluster = -1
    best_score = -1.
    for n_cluster, score in results:
        if score > best_score:
            best_cluster = n_cluster
            best_score = score

    print(best_cluster, best_score)

    return results


def get_corpus_embeddings(corpus, model_name):

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_name, device=device)
    corpus_embeddings = embedder.encode(corpus, batch_size=1024, convert_to_numpy=True, show_progress_bar=True)
    return corpus_embeddings


def main():
    parser = ArgumentParser()
    parser.add_argument("--num_clusters", default=2, type=int)
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--input_file", default="text_call.tsv", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--bsz", default=2048, type=str)
    parser.add_argument("--min_cluster", default=1000, type=int)
    parser.add_argument("--max_cluster", default=10000, type=int)
    parser.add_argument("--interval", default=1000, type=int)
    parser.add_argument("--has_silhouette", action="store_true")
    args = parser.parse_args()

    num_clusters = args.num_clusters
    num_cores = multiprocessing.cpu_count() - 4
    model_name = "beomi/kcbert-base"

    # read file
    num_tabs = 2
    utterances = set()
    with open(os.path.join(args.data_dir, args.input_file), "r", encoding="utf-8") as f:
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
                # TODO: normalize utterance
                utterances.add(utterance)

    num_clusters = min(num_clusters, len(utterances))
    print(f"num_clusters: {num_clusters}")

    corpus_embeddings = get_corpus_embeddings(corpus=list(utterances), model_name=model_name)

    sse = visualize_elbow(
        corpus_embeddings,
        num_cores,
        min_cluster=args.min_cluster,
        max_cluster=args.max_cluster,
        interval=args.interval
    )

    # plotting as fig
    _range, _sse = zip(*sse)
    plt.plot(_range, _sse, marker='o')
    plt.xlabel("n_cluster")
    plt.ylabel("sse")
    # plt.show()

    output_path = os.path.join(args.data_dir, args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, "distortion.txt"), "w", encoding="utf-8") as out_f:
        for i, _sse in sse:
            out_f.write(f"{i}\t{_sse}\n")

    plt.savefig(os.path.join(output_path, "fig.png"))

    if args.has_silhouette:
        silhouettes = visualize_silhouette_layer(
            corpus_embeddings,
            num_cores,
            min_cluster=args.min_cluster,
            max_cluster=args.max_cluster,
            interval=args.interval
        )
        with open(os.path.join(output_path, "silhouette.txt"), "w", encoding="utf-8") as out_f:
            for i, silhouette in silhouettes:
                out_f.write(f"{i}\t{silhouette}\n")


if __name__ == "__main__":
    main()
