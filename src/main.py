from phase1 import *
from math import log10

# Load data
with open("../IR_data_news_12k.json", "r") as file:
    data = json.load(file)

# Load inverted index
with open("inverted_index.json", "r") as file:
    inverted_index = json.load(file)

# Add tf-idf to inverted index
N = int(list(data.keys())[-1])  # Number of documents
for token in inverted_index:
    for doc_id in inverted_index[token]:
        if doc_id != "total_frequency":
            f_t_d = inverted_index[token][doc_id][
                "frequency"
            ]  # Frequency of token in document
            n_t = len(inverted_index[token]) - 1  # Number of documents containing token

            tf_idf = (1 + log10(f_t_d)) * log10(N / n_t)
            inverted_index[token][doc_id]["tf-idf"] = tf_idf


# print(inverted_index["جردن"])
