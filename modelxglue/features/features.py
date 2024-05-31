import json

import gensim.downloader as api
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.txt_utils import tokenizer
from .kernel_features import get_pairwise_kernel
from .transform import FeatureTransform

EMBEDDING_MODEL = 'a'
FEATURES = ('TFIDF', 'W2V', 'KERNEL', 'GRAPH', 'XMI')





def get_features_w2v(doc, model, dim=300):
    words = [w for w in tokenizer(doc) if w in model.vocab]
    if len(words) == 0:
        return np.zeros(dim)
    vectors = np.stack([model.wv[w] for w in words])
    return np.mean(vectors, axis=0)


def filter_dataset(dataset, ids):
    new_dataset = dataset[dataset['ids'].isin(ids)]
    new_dataset.ids = new_dataset.ids.astype("category")
    new_dataset.ids = new_dataset.ids.cat.set_categories(ids)
    new_dataset = new_dataset.sort_values(["ids"])
    return new_dataset


def dataset_as_format(dataset, format, ids_train, ids_val):
    format = format.upper()
    train_dataset, val_dataset = (filter_dataset(dataset, ids_train),
                                  filter_dataset(dataset, ids_val))

    if format == 'XMI':
        xmi_train = train_dataset[['ids', 'xmi']]
        xmi_val = val_dataset[['ids', 'xmi']]
        return xmi_train, xmi_val
    elif format == 'GRAPH':
        train_dataset['graph'] = train_dataset['graph'].apply(json.loads)
        val_dataset['graph'] = val_dataset['graph'].apply(json.loads)
        return train_dataset, val_dataset
    elif format == 'TXT' or format == 'TEXT':
        txt_train = train_dataset[['ids', 'txt']]
        txt_val = val_dataset[['ids', 'txt']]
        return txt_train, txt_val


def get_features(features, train_dataset, val_dataset, ids_train=None, ids_val=None):
    if ids_train is not None:
        train_dataset = filter_dataset(train_dataset, ids_train)
    if ids_val is not None:
        filter_dataset(val_dataset, ids_val)

    corpus_train = list(train_dataset['txt'])
    corpus_val = list(val_dataset['txt'])
    graphs_train = [json.loads(g) for g in train_dataset['graph']]
    graphs_val = [json.loads(g) for g in val_dataset['graph']]

    if features == 'TFIDF':
        vectorizer = TfidfVectorizer(lowercase=False, tokenizer=tokenizer, min_df=3)
        X_train = vectorizer.fit_transform(corpus_train).toarray()
        X_val = vectorizer.transform(corpus_val).toarray()
    elif features == 'W2V':
        X_train = np.array([get_features_w2v(doc, get_embedding_model()) for doc in corpus_train])
        X_val = np.array([get_features_w2v(doc, get_embedding_model()) for doc in corpus_val])
    elif features == 'XMI':
        X_train = train_dataset[['ids', 'xmi']]
        X_val = val_dataset[['ids', 'xmi']]
    elif features == 'GRAPH':
        return graphs_train, graphs_val
    elif features == 'KERNEL':
        X_train, X_val = get_pairwise_kernel(graphs_train, graphs_val, kernel='WL')
    return X_train, X_val


def get_features_recommendation(features, df_train, df_val):
    cols = [c for c in df_train.columns if c not in ['ids', 'target']]
    corpus_train = list(df_train[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1))
    corpus_val = list(df_val[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1))
    if features == 'TFIDF':
        vectorizer = TfidfVectorizer(lowercase=False, tokenizer=tokenizer, min_df=3)
        X_train = vectorizer.fit_transform(corpus_train).toarray()
        X_val = vectorizer.transform(corpus_val).toarray()
    elif features == 'W2V':
        X_train = np.array([get_features_w2v(doc, get_embedding_model()) for doc in corpus_train])
        X_val = np.array([get_features_w2v(doc, get_embedding_model()) for doc in corpus_val])
    elif features == 'XMI':
        X_train = corpus_train[['ids', 'xmi']]
        X_val = corpus_val[['ids', 'xmi']]
    return X_train, X_val
