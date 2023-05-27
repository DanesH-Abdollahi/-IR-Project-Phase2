import json
from hazm import Normalizer, WordTokenizer, Lemmatizer, SentenceTokenizer
from hazm.utils import stopwords_list

with open("../IR_data_news_12k.json", "r") as file:
    data = json.load(file)

normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmetizer = Lemmatizer()
tokens_dict = dict()
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
        '》',
        '《',
        "**",
        '*',
        '**',
        '***',
        '****',
        '********',

    ]
)

for id in data:
    tmp_tokens = []
    tokens_dict[id] = normalizer.normalize(
        data[id]["content"])  # normalize text
    tmp_tokens = tokenizer.tokenize(tokens_dict[id])  # list of tokens

    tmp_tokens = [
        token for token in tmp_tokens if token not in stopwords
    ]  # remove stopwords

    for i in range(len(tmp_tokens)):
        tmp_tokens[i] = lemmetizer.lemmatize(tmp_tokens[i])  # lemmatize tokens

    # update tokens_dict with lemmatized tokens list
    tokens_dict[id] = tmp_tokens


# Construct inverted index
inverted_index = dict()

for id in tokens_dict:
    for token in enumerate(tokens_dict[id]):
        if token[1] not in inverted_index:
            inverted_index[token[1]] = dict()
            inverted_index[token[1]]["total_frequency"] = 1
            inverted_index[token[1]][id] = dict()
            inverted_index[token[1]][id]["frequency"] = 1
            inverted_index[token[1]][id]["positions"] = [token[0]]

        else:
            inverted_index[token[1]]["total_frequency"] += 1
            if id not in inverted_index[token[1]]:
                inverted_index[token[1]][id] = dict()
                inverted_index[token[1]][id]["frequency"] = 1
                inverted_index[token[1]][id]["positions"] = [token[0]]
            else:
                inverted_index[token[1]][id]["frequency"] += 1
                inverted_index[token[1]][id]["positions"].append(token[0])

# Sort inverted index by token
sorted_inverted_index = dict(
    sorted(inverted_index.items(), key=lambda item: item[0]))

# Query processing
query = input("Enter your query : ")
query = normalizer.normalize(query)
query = tokenizer.tokenize(query)
query = [token for token in query if token not in stopwords or token ==
         "!" or token == '«' or token == '»']
for i in range(len(query)):
    query[i] = lemmetizer.lemmatize(query[i])

# remove tokens that are not in inverted index to avoid key error
query = [token for token in query if token in sorted_inverted_index.keys(
) or token == "!" or token == '«' or token == '»']

and_tokens = []
not_tokens = []
and_phrase = []
not_phrase = []

not_flag = 0
phrase_flag = 0
skip_flag = 0
for i in range(len(query)):
    if skip_flag:
        skip_flag -= 1
        continue

    if query[i] == '!':
        not_flag = 1
        continue

    if query[i] == '«':
        phrase_flag = 1
        continue

    if not_flag == 1:
        if phrase_flag == 1:
            skip_flag = query.index('»', i) - i
            not_phrase.append(query[i: i + skip_flag])
            phrase_flag = 0
            not_flag = 0
            continue
        not_tokens.append(query[i])
        not_flag = 0
        continue

    if phrase_flag == 1:
        skip_flag = query.index('»', i) - i
        and_phrase.append(query[i: i + skip_flag])
        phrase_flag = 0
        continue

    and_tokens.append(query[i])


intersection = set()
union = set()
refrence = set(data.keys())

if and_tokens:
    for i in range(len(and_tokens)):
        if i == 0:
            intersection = set(sorted_inverted_index[and_tokens[i]].keys())
            union = intersection
            continue

        intersection = intersection.intersection(
            set(sorted_inverted_index[and_tokens[i]].keys()))
        union = union.union(
            set(sorted_inverted_index[and_tokens[i]].keys()))

if not_tokens:
    for i in range(len(not_tokens)):
        if not intersection:
            intersection = refrence - \
                set(sorted_inverted_index[not_tokens[i]].keys())
            if not union:
                union = intersection
                continue
            union = union.union(intersection)
            continue

        intersection = intersection.intersection(
            refrence - set(sorted_inverted_index[not_tokens[i]].keys()))
        union = union.union(
            refrence - set(sorted_inverted_index[not_tokens[i]].keys()))


if and_phrase:
    for phrase in and_phrase:
        phrase_intersection = set()
        for i in range(len(phrase)):
            if i == 0:
                phrase_intersection = set(
                    sorted_inverted_index[phrase[i]].keys())
                continue

            phrase_intersection = phrase_intersection.intersection(
                set(sorted_inverted_index[phrase[i]].keys()))

        phrase_intersection = phrase_intersection - set(['total_frequency'])

        tmp_phrase_intersection = phrase_intersection.copy()
        for id in tmp_phrase_intersection:
            tmp = sorted_inverted_index[phrase[0]][id]["positions"]
            for j in range(1, len(phrase)):
                tmp = [x for x in tmp if x +
                       j in sorted_inverted_index[phrase[j]][id]["positions"]]

            if not tmp:
                phrase_intersection.remove(id)

        if not intersection:
            intersection = phrase_intersection
            if not union:
                union = phrase_intersection
                continue
            continue

        intersection = intersection.intersection(phrase_intersection)
        union = union.union(phrase_intersection)


if not_phrase:
    for phrase in not_phrase:
        phrase_intersection = set()
        for i in range(len(phrase)):
            if i == 0:
                phrase_intersection = set(
                    sorted_inverted_index[phrase[i]].keys())
                continue

            phrase_intersection = phrase_intersection.intersection(
                set(sorted_inverted_index[phrase[i]].keys()))

        phrase_intersection = phrase_intersection - set(['total_frequency'])

        tmp_phrase_intersection = phrase_intersection.copy()
        for id in tmp_phrase_intersection:
            tmp = sorted_inverted_index[phrase[0]][id]["positions"]
            for j in range(1, len(phrase)):
                tmp = [x for x in tmp if x +
                       j in sorted_inverted_index[phrase[j]][id]["positions"]]

            if not tmp:
                phrase_intersection.remove(id)

        if not intersection:
            intersection = refrence - phrase_intersection
            if not union:
                union = intersection
                continue
            union = union.union(intersection)
            continue

        intersection = intersection.intersection(
            refrence - phrase_intersection)
        union = union.union(refrence - phrase_intersection)

# remove total_frequency from union
intersection = intersection - set(['total_frequency'])
union = union - set(['total_frequency'])  # remove total_frequency from union
union = union - intersection  # remove intersection from union

intersection = sorted(intersection, key=lambda x: int(x))
union = sorted(union, key=lambda x: int(x))
query_result = intersection + union

print(intersection)
print(len(intersection))
print("ـ"*50)
print(union)
print(len(union))
print("ـ"*50)
print(query_result)
print(len(query_result))

sentence_tokenizer = SentenceTokenizer()

if len(query_result) > 5:
    query_result_for_show = query_result[0:5]


sentences = dict()
for id in query_result_for_show:
    content = data[id]['content']
    content = normalizer.normalize(content)
    sentences[id] = sentence_tokenizer.tokenize(content)
    tmp = sentences[id].copy()
    for sent in tmp:
        tokens = tokenizer.tokenize(sent)
        tokens = [lemmetizer.lemmatize(token) for token in tokens]

        flag = 0
        for query_token in and_tokens:
            if query_token in tokens:
                flag = 1
                break

        for phrase in and_phrase:
            phrase_flag = 0
            if phrase[0] in tokens:
                phrase_flag += 1
                place = tokens.index(phrase[0])
                for i in range(1, len(phrase)):
                    if phrase[i] == tokens[place + i]:
                        phrase_flag += 1

            if phrase_flag == len(phrase):
                flag = 1
                break

        if flag == 0:
            sentences[id].remove(sent)

    sentences[id] = "".join(sentences[id])


for id in query_result_for_show:
    print(f"Document ID :{int(id):< 6}")
    print(f"Title : {data[id]['title']}")
    print(f"Related Content : {sentences[id]}")
    print("ـ"*100)
