import json
from hazm import Normalizer, WordTokenizer, Lemmatizer, SentenceTokenizer
from hazm.utils import stopwords_list
from math import log10

# Initialize hazm tools
normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmetizer = Lemmatizer()
sentence_tokenizer = SentenceTokenizer()
stopwords = stopwords_list()
stopwords.extend(
    [
        "،",
        ".",
        ":",
        "؛",
        "!",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "«",
        "»",
        "؟",
        "-",
        "/",
        '"',
        "'",
        "*",
        "!!",
        "!؟",
        "''",
        '""',
        "》",
        "《",
        "**",
        "*",
        "**",
        "***",
        "****",
        "********",
    ]
)

# Load data from file
with open("../IR_data_news_12k.json", "r") as file:
    data = json.load(file)

# Load Inverted Index from file (which was created in phase 1)
with open("inverted_index.json", "r") as file:
    inverted_index = json.load(file)


tokens_dict = dict()
for id in data:
    tmp = normalizer.normalize(data[id]["content"])  # normalize text
    tmp = tokenizer.tokenize(tmp)  # list of tokens

    tmp = [token for token in tmp if token not in stopwords]  # remove stopwords
    tmp = [lemmetizer.lemmatize(token) for token in tmp]

    tokens_dict[id] = set(tmp)

# -------------------------------------------------------------------------------------------------

# Add tf-idf to inverted index
document_norms = {}
N = int(list(data.keys())[-1])  # Number of documents
for token in inverted_index:
    n_t = len(inverted_index[token]) - 1  # Number of documents containing token
    for doc_id in inverted_index[token]:
        if doc_id != "total_frequency":
            f_t_d = inverted_index[token][doc_id][
                "frequency"
            ]  # Frequency of token in document
            tf_idf = (1 + log10(f_t_d)) * log10(N / n_t)
            inverted_index[token][doc_id]["tf-idf"] = tf_idf

            if doc_id not in document_norms:
                document_norms[doc_id] = tf_idf**2
            else:
                document_norms[doc_id] += tf_idf**2

# Normalize Documents vector
for token in inverted_index:
    for doc_id in inverted_index[token]:
        if doc_id != "total_frequency" and doc_id != "champion_list":
            inverted_index[token][doc_id]["tf-idf"] /= document_norms[doc_id] ** 0.5

# -------------------------------------------------------------------------------------------------

# Add champion list to inverted index for each token
r = 10  # Number of documents in champion list
for token in inverted_index:
    champion_list = []
    for doc_id in inverted_index[token]:
        if doc_id != "total_frequency" and doc_id != "champion_list":
            champion_list.append((doc_id, inverted_index[token][doc_id]["tf-idf"]))

    champion_list = sorted(champion_list, key=lambda x: x[1], reverse=True)
    inverted_index[token]["champion_list"] = champion_list[:r]

# -------------------------------------------------------------------------------------------------
K = 5  # Number of documents to retrieve

# Query Processing
query = input("Enter your query: ")
query = normalizer.normalize(query)
query = tokenizer.tokenize(query)
query = [token for token in query if token not in stopwords]
query = [lemmetizer.lemmatize(token) for token in query]

# Remove tokens that are not in inverted index
for token in query:
    if token not in inverted_index:
        query.remove(token)

# Construct query vector
query_vector = {}
for token in query:
    if token not in query_vector:
        query_vector[token] = 1
    else:
        query_vector[token] += 1

total_abs = 0
for token in query_vector:
    n_t = len(inverted_index[token]) - 1  # Number of documents containing token
    t_f = query_vector[token]
    query_vector[token] = (1 + log10(t_f)) * log10(N / n_t)
    total_abs += query_vector[token] ** 2

total_abs = total_abs**0.5

# Normalize the query vector
for token in query_vector:
    query_vector[token] /= total_abs

# -------------------------------------------------------------------------------------------------

# Calculate cosine similarity
cosine_similarity = {}
for token in query_vector:
    for doc_id in inverted_index[token]:
        if doc_id != "total_frequency" and doc_id != "champion_list":
            tf_idf = inverted_index[token][doc_id]["tf-idf"]
            if doc_id not in cosine_similarity:
                cosine_similarity[doc_id] = tf_idf * query_vector[token]
            else:
                cosine_similarity[doc_id] += tf_idf * query_vector[token]


# Sort documents based on cosine similarity
cosine_similarity_results = sorted(
    cosine_similarity.items(), key=lambda x: x[1], reverse=True
)

# -------------------------------------------------------------------------------------------------

# Write top K results for cosine similarity in Results.txt
results_for_show = cosine_similarity_results[:K]
sentences = dict()
for id, _ in results_for_show:
    content = data[id]["content"]
    content = normalizer.normalize(content)
    sentences[id] = sentence_tokenizer.tokenize(content)
    tmp = sentences[id].copy()
    for sent in tmp:
        tokens = tokenizer.tokenize(sent)
        tokens = [lemmetizer.lemmatize(token) for token in tokens]

        flag = 0
        for query_token in query:
            if query_token in tokens:
                flag = 1
                break

        if flag == 0:
            sentences[id].remove(sent)

with open("Results.txt", "w") as file:
    file.write("********** Results for Cosine Similarity **********\n\n")
    for id, _ in results_for_show:
        file.write(f"Document ID: {id}\n")
        file.write(f"Document Title: {data[id]['title']}\n")
        file.write("Related Sentences:\n")
        for sent in sentences[id]:
            file.write(f"{sent}\n")
        file.write("-" * 100 + "\n\n")

# -------------------------------------------------------------------------------------------------

# Calculate Cosine Similarity with champion list method
cosine_similarity_champion_list = {}
champion_list = []
for token in query_vector:
    for doc_id, _ in inverted_index[token]["champion_list"]:
        if doc_id not in champion_list:
            champion_list.append(doc_id)

for doc_id in champion_list:
    cosine_similarity_champion_list[doc_id] = 0
    for token in query_vector:
        if doc_id in inverted_index[token]:
            tf_idf = inverted_index[token][doc_id]["tf-idf"]
            cosine_similarity_champion_list[doc_id] += tf_idf * query_vector[token]


# Sort documents based on cosine similarity for champion list
cosine_similarity_champion_list_results = sorted(
    cosine_similarity_champion_list.items(), key=lambda x: x[1], reverse=True
)


# -------------------------------------------------------------------------------------------------

# Write top K results for cosine similarity with champion list in Results.txt
results_for_show = cosine_similarity_champion_list_results[:K]
sentences = dict()

for id, _ in results_for_show:
    content = data[id]["content"]
    content = normalizer.normalize(content)
    sentences[id] = sentence_tokenizer.tokenize(content)
    tmp = sentences[id].copy()
    for sent in tmp:
        tokens = tokenizer.tokenize(sent)
        tokens = [lemmetizer.lemmatize(token) for token in tokens]

        flag = 0
        for query_token in query:
            if query_token in tokens:
                flag = 1
                break

        if flag == 0:
            sentences[id].remove(sent)

with open("Results.txt", "a") as file:
    file.write(
        "********** Results for Cosine Similarity with Champion List **********\n\n"
    )
    for id, _ in results_for_show:
        file.write(f"Document ID: {id}\n")
        file.write(f"Document Title: {data[id]['title']}\n")
        file.write("Related Sentences:\n")
        for sent in sentences[id]:
            file.write(f"{sent}\n")
        file.write("-" * 100 + "\n\n")

# -------------------------------------------------------------------------------------------------

# Calculate jaccard similarity using inverted index and data collection
jaccard_similarity = {}
for token in query_vector:
    for doc_id in inverted_index[token]:
        if doc_id != "total_frequency" and doc_id != "champion_list":
            if doc_id not in jaccard_similarity:
                intersection_length = len(tokens_dict[doc_id].intersection(set(query)))
                union_length = len(tokens_dict[doc_id].union(set(query)))
                jaccard_similarity[doc_id] = intersection_length / union_length

# Sort documents based on jaccard similarity
jaccard_similarity_result = sorted(
    jaccard_similarity.items(), key=lambda x: x[1], reverse=True
)

# -------------------------------------------------------------------------------------------------

# Write top K results for jaccard similarity in Results.txt
results_for_show = jaccard_similarity_result[:K]
sentences = dict()
for id, _ in results_for_show:
    content = data[id]["content"]
    content = normalizer.normalize(content)
    sentences[id] = sentence_tokenizer.tokenize(content)
    tmp = sentences[id].copy()
    for sent in tmp:
        tokens = tokenizer.tokenize(sent)
        tokens = [lemmetizer.lemmatize(token) for token in tokens]

        flag = 0
        for query_token in query:
            if query_token in tokens:
                flag = 1
                break

        if flag == 0:
            sentences[id].remove(sent)

with open("Results.txt", "a") as file:
    file.write("********** Results for Jaccard Similarity **********\n\n")
    for id, _ in results_for_show:
        file.write(f"Document ID: {id}\n")
        file.write(f"Document Title: {data[id]['title']}\n")
        file.write("Related Sentences:\n")
        for sent in sentences[id]:
            file.write(f"{sent}\n")
        file.write("-" * 100 + "\n\n")

# -------------------------------------------------------------------------------------------------

# Calculate Jaccard Similarity with champion list method
jaccard_similarity_champion_list = {}
champion_list = []
for token in query_vector:
    for doc_id, _ in inverted_index[token]["champion_list"]:
        if doc_id not in champion_list:
            champion_list.append(doc_id)


for doc_id in champion_list:
    intersection_length = len(tokens_dict[doc_id].intersection(set(query)))
    union_length = len(tokens_dict[doc_id].union(set(query)))
    jaccard_similarity_champion_list[doc_id] = intersection_length / union_length

# Sort documents based on jaccard similarity for champion list
jaccard_similarity_champion_list_results = sorted(
    jaccard_similarity_champion_list.items(), key=lambda x: x[1], reverse=True
)

# -------------------------------------------------------------------------------------------------

# Write top K results for jaccard similarity with champion list in Results.txt
results_for_show = jaccard_similarity_champion_list_results[:K]
sentences = dict()
for id, _ in results_for_show:
    content = data[id]["content"]
    content = normalizer.normalize(content)
    sentences[id] = sentence_tokenizer.tokenize(content)
    tmp = sentences[id].copy()
    for sent in tmp:
        tokens = tokenizer.tokenize(sent)
        tokens = [lemmetizer.lemmatize(token) for token in tokens]

        flag = 0
        for query_token in query:
            if query_token in tokens:
                flag = 1
                break

        if flag == 0:
            sentences[id].remove(sent)

with open("Results.txt", "a") as file:
    file.write(
        "********** Results for Jaccard Similarity with Champion List **********\n\n"
    )
    for id, _ in results_for_show:
        file.write(f"Document ID: {id}\n")
        file.write(f"Document Title: {data[id]['title']}\n")
        file.write("Related Sentences:\n")
        for sent in sentences[id]:
            file.write(f"{sent}\n")
        file.write("-" * 100 + "\n\n")
