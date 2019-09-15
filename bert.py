from pprint import pprint #prettyprint - makes the output easier to read
from nltk.corpus import stopwords # To remove stopwords
import pandas as pd
from bert import tokenization
from bert import modeling
import numpy as np
import bert_funcs
from sklearn.cluster import KMeans
from tqdm import tqdm
import re
pd.set_option('display.max_columns', None)
nltk.download('stopwords')
stopwords = set(stopwords.words("english"))

with open('reviews_as_raw_text.txt') as fopen:
    negative = fopen.read().split('\n')[:-1]
negative = [re.sub(r'[^\w\s]','',str(item)) for item in negative]
test = []
for x in negative:
    if len(x) < 500:
        test.append(x)

negative = test
##bert setup
BERT_VOCAB = 'cased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'cased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = 'cased_L-12_H-768_A-12/bert_config.json'


tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=False)
bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
model = bert_funcs._Model(bert_config,tokenizer)

batch_size = 5
ngram = (1,3)
n_topics = 3
rows, attentions = [], []
for i in tqdm(range(0, len(negative), batch_size)):
    index = min(i + batch_size, len(negative))
    rows.append(model.vectorize(negative[i:index]))
    attentions.extend(model.attention(negative[i:index]))


concat = np.concatenate(rows, axis = 0)
kmeans = KMeans(n_clusters = n_topics, random_state = 0).fit(concat)
labels = kmeans.labels_

overall, filtered_a = [], []
for a in attentions:
    f = [i for i in a if i[0] not in stopwords]
    overall.extend(f)
    filtered_a.append(f)

o_ngram = bert_funcs.generate_ngram(overall, ngram)
features = []
for i in o_ngram:
    features.append(' '.join([w[0] for w in i]))
features = list(set(features))

components = np.zeros((n_topics, len(features)))
for no, i in enumerate(labels):
    if (no + 1) % 500 == 0:
        print('processed %d'%(no + 1))
        print('processed asd')
    f = bert_funcs.generate_ngram(filtered_a[no], ngram)
    for w in f:
        word = ' '.join([r[0] for r in w])
        score = np.mean([r[1] for r in w])
        if word in features:
            components[i, features.index(word)] += score



pprint(bert_funcs.print_topics_modelling(
        n_topics,
        feature_names=np.array(features),
        sorting=np.argsort(components)[:, ::-1],
        n_words=30,
        return_df=True,
    ))
