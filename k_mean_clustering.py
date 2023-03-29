from KoBERTScore import BERTScore
from sklearn.cluster import KMeans
from transformers import BertModel, AutoModel, AutoTokenizer

def main():
    # model_name = "monologg/koelectra-base-v2-discriminator"
    model_name = "beomi/kcbert-base"
    bert_score = BERTScore(model_name, best_layer=12)

    references = [
        "배고파",
        "배고파",
        "배고파",
        "배고파",
    ]
    candidates = [
        "배고파",
        "아 배고프다",
        "아 배아파",
        "아 배고프다",
    ]

    scores = bert_score(references, candidates, retrain_idf=False, batch_size=128)
    print(scores)
#     scores = bert_score(references, candidates, retrain_idf=True, batch_size=128)
#     print(scores)

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(["너는 이름이 뭐야", "나는 김말이야"], return_tensors="pt")
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    print(len(outputs))
    pooler_output = outputs["pooler_output"]
    print(pooler_output)
    print(pooler_output.size())


def get_cluster_kmeans():

    num_clusters = 2
    model_name = "beomi/kcbert-base"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Corpus with example sentences
    corpus = [
        "배고파",
        "아 배고프다",
        "아 배아파",
        "아 배가 너무 아프다",
              ]
    inputs = tokenizer(corpus, padding=True, return_tensors="pt")
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    pooler_output = outputs["pooler_output"]
    # corpus_embeddings = pooler_output

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name)
    corpus_tensor = embedder.encode(corpus, convert_to_numpy=False)
    corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True)

    # clustering_model = KMeans(n_clusters=num_clusters, random_state=random_state)
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)

    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i+1)
        print(cluster)
        print("")


if __name__ == "__main__":
    # main()
    get_cluster_kmeans()
