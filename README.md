# Ranked Retrieval

This Python code is an implementation of an Information Retrieval system that performs document retrieval based on query similarity using various methods. Here is a breakdown of the code's functionality:

## 1. Importing Libraries

The code starts by importing necessary libraries such as `json` for working with JSON files, `hazm` for Persian text processing, `math` for mathematical calculations, and others.

## 2. Initializing Hazm Tools

Several Hazm tools are initialized, including a normalizer, tokenizer, lemmatizer, sentence tokenizer, and a list of stopwords. These tools are used for preprocessing text data.

## 3. Loading Data

The code loads data from a JSON file (`IR_data_news_12k.json`) that contains information about documents to be searched.

## 4. Loading Inverted Index

The code loads the inverted index from a JSON file (`inverted_index.json`) that was created in a previous phase of the Information Retrieval system.

## 5. Preprocessing Documents

The code preprocesses the documents by normalizing the text, tokenizing it into words, removing stopwords, and lemmatizing the tokens. The preprocessed tokens are stored in a dictionary (`tokens_dict`) with document IDs as keys.

## 6. Adding TF-IDF to Inverted Index

The code calculates the TF-IDF (Term Frequency-Inverse Document Frequency) for each token in the inverted index. TF-IDF is a numerical statistic used to reflect the importance of a word in a document relative to a collection of documents. The TF-IDF values are added to the inverted index.

## 7. Normalizing Document Vectors

The code normalizes the document vectors in the inverted index by dividing the TF-IDF values by the square root of the sum of squares of the TF-IDF values for each document.

## 8. Creating Champion Lists

The code creates a champion list for each token in the inverted index. The champion list contains a limited number of top documents based on their TF-IDF values.

## 9. Query Processing

The code prompts the user to enter a query and processes the query by normalizing, tokenizing, removing stopwords, and lemmatizing the query tokens.

## 10. Constructing Query Vector

The code constructs a query vector based on the query tokens. The query vector contains the TF-IDF values for each token in the query.

## 11. Calculating Cosine Similarity

The code calculates the cosine similarity between the query vector and the document vectors in the inverted index. Cosine similarity is a measure of similarity between two non-zero vectors.

## 12. Sorting and Writing Results for Cosine Similarity

The code sorts the documents based on their cosine similarity scores and writes the top K results to a file named "Results.txt". The related sentences are extracted from the documents and included in the results.

## 13. Calculating Cosine Similarity with Champion List

The code calculates the cosine similarity between the query vector and the document vectors in the champion lists of the inverted index. The champion lists contain a limited number of top documents based on their TF-IDF values.

## 14. Sorting and Writing Results for Cosine Similarity with Champion List

The code sorts the documents based on their cosine similarity scores using the champion lists and writes the top K results to "Results.txt". The related sentences are extracted from the documents and included in the results.

## 15. Calculating Jaccard Similarity

The code calculates the Jaccard similarity between the query tokens and the tokens in each document using the inverted index. Jaccard similarity is a measure of similarity between two sets.

## 16. Sorting and Writing Results for Jaccard Similarity

The code sorts the documents based on their Jaccard similarity scores and writes the top K results to "Results.txt". The related sentences are extracted from the documents and included in the results.

## 17. End of Code Execution

The code execution ends after writing the results to the file.

