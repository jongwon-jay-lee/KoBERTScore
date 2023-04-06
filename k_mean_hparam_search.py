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


def visualize_elbow(corpus_embeddings, num_cores, min_cluster=1000, max_cluster=5000):
    distortions = []
    for i in range(min_cluster, max_cluster):
        clustering_model = MiniBatchKMeans(n_clusters=i, verbose=1, batch_size=256 * num_cores)
        clustering_model.fit(corpus_embeddings)
        distortions.append(clustering_model.inertia_)
    return distortions


def get_cluster_kmeans(corpus, num_cores, model_name):

    device = f"cuda" if torch.cuda.is_available() else "cpu"

    embedder = SentenceTransformer(model_name, device=device)
    corpus_embeddings = embedder.encode(corpus, batch_size=1024, convert_to_numpy=True, show_progress_bar=True)

    distortions = visualize_elbow(corpus_embeddings, num_cores)
    return distortions


def main():
    parser = ArgumentParser()
    parser.add_argument("--num_clusters", default=2, type=int)
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--input_file", default="text_call.tsv", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
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
    distortions = get_cluster_kmeans(
        corpus=list(utterances), num_cores=num_cores, model_name=model_name
    )

    output_path = os.path.join(args.data_dir, args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, "distortion.txt"), "w", encoding="utf-8") as out_f:
        for distortion in distortions:
            out_f.write(f"{distortion}\n")


if __name__ == "__main__":
    main()
