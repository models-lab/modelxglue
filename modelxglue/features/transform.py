import os
import os.path
import copy

import pandas as pd
from grakel import WeisfeilerLehman
from sklearn.feature_extraction.text import TfidfVectorizer

from ..features.kernel_features import to_grakel_graph
from ..utils.txt_utils import tokenizer

import numpy as np

TRAIN = 'training-set'
TEST = 'test-set'


class TransformConfiguration:

    def __init__(self):
        self.by_type = {}

    def add(self, what, transform):
        if what == 'all':
            self.by_type[TRAIN] = transform.for_(TRAIN)
            self.by_type[TEST] = transform.for_(TEST)
            return

        what = TRAIN if what in ['train', 'training'] else what
        what = TEST if what in ['test', 'testing'] else what
        self.by_type[what] = transform.for_(what)

    def get_transform_for(self, *transform_phases):
        ctx = TransformContext()
        res = []
        for what in transform_phases:
            if what in self.by_type:
                t = copy.copy(self.by_type[what])
            else:
                t = NoneFeatureTransform()
            t.set_context(ctx)
            res.append(t)
        return res


class TransformContext(dict):
    pass


class FeatureTransform(object):
    """A class that represents a feature transform. """

    def __init__(self):
        self.phase = None
        self.ctx = None

    def transform(self, df, what):
        raise NotImplementedError

    def set_context(self, ctx):
        self.ctx = ctx

    def for_(self, what):
        f = copy.copy(self)
        f.phase = what
        return f


class CompositeTransform(FeatureTransform):

    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def transform(self, df, what):
        for t in self.steps:
            df = t.transform(df, what)
        return df

    # Override for_ and set_phase to propagate to all steps
    def for_(self, what):
        f = super().for_(what)
        f.steps = [s.for_(what) for s in f.steps]
        return f

    def set_context(self, ctx):
        super().set_context(ctx)
        for s in self.steps:
            s.set_context(ctx)

class NoneFeatureTransform(FeatureTransform):

    def transform(self, df, what):
        return df


class DumpXmiTransform(FeatureTransform):

    def __init__(self, cache, model_type):
        super().__init__()
        self.cache_dir = cache
        self.model_type = model_type

    def transform(self, df, what):
        xmi_folder = os.path.join("xmi-dump", what)
        full_xmi_folder = os.path.join(self.cache_dir, xmi_folder)
        os.makedirs(full_xmi_folder, exist_ok=True)

        for index, row in df.iterrows():
            extension = os.path.splitext(row['ids'])[1]
            if extension is None or extension == '':
                extension = '.' + self.model_type
            xmi_contents = row['xmi']
            xmi_file = os.path.join(full_xmi_folder, f"file-{index}{extension}")
            with open(xmi_file, 'w') as f:
                f.write(xmi_contents)

            df.loc[index, 'xmi_path'] = os.path.join(xmi_folder, f"file-{index}{extension}")

        # drop xmi column from df
        df = df.drop(columns=['xmi'])
        df.attrs['xmi_folder'] = xmi_folder
        return df


class VectorizeText(FeatureTransform):

    def __init__(self, columns, strategy, separator=''):
        super().__init__()
        self.columns = columns
        self.strategy = strategy
        self.embedding_model = None
        self.separator = "\n" if separator == "\\n" or separator == 'newline' else separator

    def transform(self, df, what):
        df.reset_index(drop=True, inplace=True)
        cols = [c for c in df.columns if c in self.columns]
        corpus = list(df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1))
        if self.strategy.lower() == 'tfidf':
            if self.phase == TRAIN:
                vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda doc: tokenizer(doc, self.separator), min_df=3)
                as_vector = vectorizer.fit_transform(corpus).toarray()
                self.ctx['vectorizer'] = vectorizer
            else:
                vectorizer = self.ctx['vectorizer']
                as_vector = vectorizer.transform(corpus).toarray()

            # Be smarter here and check that everything is correctly set in the YAML
        elif self.strategy.lower() == 'glove':
            as_vector = np.array([self.get_features_w2v(doc, self.get_embedding_model()) for doc in corpus])
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        df_vector = pd.DataFrame(as_vector)
        return pd.concat([df, df_vector], axis=1)

    def get_features_w2v(self, doc, model, dim=300):
        words = [w for w in tokenizer(doc, separator=self.separator) if w in model.vocab]
        if len(words) == 0:
            return np.zeros(dim)
        vectors = np.stack([model.wv[w] for w in words])
        return np.mean(vectors, axis=0)

    def get_embedding_model(self):
        if self.embedding_model is None:
            import gensim.downloader as api
            self.embedding_model = api.load("glove-wiki-gigaword-300")
        return self.embedding_model


class KernelTransform(FeatureTransform):

    def __init__(self, column):
        super().__init__()
        self.column = column

    def transform(self, df, what):
        df.reset_index(drop=True, inplace=True)
        graphs = list(df[self.column])

        if self.phase == TRAIN:
            G_train = [to_grakel_graph(g) for g in graphs]
            # This can be parameterized
            kernel = WeisfeilerLehman(n_iter=3)
            as_vector = kernel.fit_transform(G_train)
            self.ctx['kernel'] = kernel
        else:
            G_test = [to_grakel_graph(g) for g in graphs]
            kernel = self.ctx['kernel']
            as_vector = kernel.transform(G_test)

        df_vector = pd.DataFrame(as_vector)
        return pd.concat([df, df_vector], axis=1)