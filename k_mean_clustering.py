import os
import re
import torch
import multiprocessing
from KoBERTScore import BERTScore
from sklearn.cluster import KMeans, MiniBatchKMeans
from transformers import BertModel, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser


ALLOCATED_GPUS = None


def get_cluster_kmeans(corpus, num_clusters, num_cores, model_name):
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]:
        global ALLOCATED_GPUS
        ALLOCATED_GPUS = str(os.environ["CUDA_VISIBLE_DEVICES"]).split(",")[0]
        ALLOCATED_GPUS = ":" + ALLOCATED_GPUS
    else:
        ALLOCATED_GPUS = ""
    device = f"cuda{ALLOCATED_GPUS}" if torch.cuda.is_available() else "cpu"
    """
    device = f"cuda" if torch.cuda.is_available() else "cpu"

    embedder = SentenceTransformer(model_name, device=device)
    corpus_embeddings = embedder.encode(corpus, batch_size=1024, convert_to_numpy=True, show_progress_bar=True)

    # clustering_model = KMeans(n_clusters=num_clusters, random_state=random_state)
    # clustering_model = KMeans(n_clusters=num_clusters, verbose=1)
    clustering_model = MiniBatchKMeans(n_clusters=num_clusters, verbose=1, batch_size=256*num_cores)
    print(f"Start clustering... with {num_cores} cores")
    clustering_model.fit(corpus_embeddings)

    cluster_assignment = clustering_model.labels_

    print("End clustering...")

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
    parser.add_argument("--output_dir", default="outputs", type=str)
    args = parser.parse_args()

    num_clusters = args.num_clusters
    num_cores = multiprocessing.cpu_count() - 4
    model_name = "beomi/kcbert-base"

    # read file
    num_tabs = 2
    utterances = set()
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
                # TODO: normalize utterance
                utterances.add(utterance)

    num_clusters = min(num_clusters, len(utterances))
    print(f"num_clusters: {num_clusters}")
    clustered_sentences = get_cluster_kmeans(
        corpus=list(utterances), num_clusters=num_clusters, num_cores=num_cores, model_name=model_name
    )

    output_path = os.path.join(args.data_dir, args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for c_idx in range(num_clusters):
        print(f"{c_idx} / {num_clusters}")
        with open(os.path.join(output_path, f"{c_idx}.txt"), "w", encoding="utf-8") as out_f:
            for sent in clustered_sentences[c_idx]:
                out_f.write(f"{sent}\n")


if __name__ == "__main__":
    main()
