import random

from sklearn.model_selection import train_test_split


class SplitConfiguration:
    def __init__(self, dataset, seed, train_split, test_split, val_split, test_dataset_df=None):
        if test_dataset_df is not None:
            self.ids_train_val = list(dataset['ids'])
            self.ids_train, self.ids_val = train_test_split(list(dataset['ids']),
                                                            test_size=val_split,
                                                            random_state=seed)

            self.ids_test = list(test_dataset_df['ids'])
            self.test_dataset = test_dataset_df

            if 0 < test_split < 1:
                random.seed(123)
                self.ids_tests = random.sample(self.ids_test, k=int(test_split * len(self.ids_test)))
                self.test_dataset = self.test_dataset.loc[self.test_dataset['ids'].isin(self.ids_tests)]
        else:
            self.ids_train_val, self.ids_test = train_test_split(list(dataset['ids']),
                                                                 test_size=test_split,
                                                                 random_state=seed)
            self.ids_train, self.ids_val = train_test_split(self.ids_train_val,
                                                            test_size=val_split / train_split,
                                                            random_state=seed)

            self.test_dataset = dataset.loc[dataset['ids'].isin(self.ids_test)]

        self.train_val_dataset = dataset.loc[dataset['ids'].isin(self.ids_train_val)]
        self.train_dataset = dataset.loc[dataset['ids'].isin(self.ids_train)]
        self.val_dataset = dataset.loc[dataset['ids'].isin(self.ids_val)]

        assert len(self.train_val_dataset) > 0
        assert len(self.train_dataset) > 0
        assert len(self.val_dataset) > 0
        assert len(self.test_dataset) > 0
