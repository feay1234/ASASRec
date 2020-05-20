from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import pandas as pd
import random

class RawDataset():
    def __init__(self, df):
        # pre-process
        df = (df
              .merge(df.groupby('uid').iid.nunique().reset_index().rename(columns={'iid': 'num_uniq_vid'}),
                     on='uid', how='left')
              .merge(df.groupby('iid').uid.nunique().reset_index().rename(columns={'uid': 'num_uniq_uid'}),
                     on='iid', how='left'))
        df = df[(df.num_uniq_vid >= 10) & ((df.num_uniq_uid >= 10))]

        # index start at one and index zero is used for masking
        df.uid = df.uid.astype('category').cat.codes.values + 1
        df.iid = df.iid.astype('category').cat.codes.values + 1

        uNum = df.uid.nunique()
        iNum = df.iid.nunique()
        df.sort_values(["uid", "timestamp"], inplace=True)
        self.testRatings = df.groupby("uid").tail(1)[["uid", "iid"]].values.tolist()
        # for each user, remove last interaction from training set
        df = df.groupby("uid", as_index=False).apply(lambda x: x.iloc[:-1])

        mat = sp.dok_matrix((uNum + 1, iNum + 1), dtype=np.float32)
        seq = defaultdict(list)

        for u, i in df[["uid", "iid"]].values.tolist():
            mat[u, i] = 1.0
            seq[u].append(i)

        self.trainMatrix = mat
        self.trainSeq = seq
        self.df = df

        random.seed(2019)
        candidates = df.iid.tolist()

        negatives = []
        for u in range(uNum):
            neg = []
            for i in range(100):
                r = random.choice(candidates)
                while (u, r) in mat or self.testRatings[u] == r:
                    r = random.choice(candidates)
                neg.append(r)
            negatives.append(neg)

        self.testNegatives = negatives
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape


class Dataset():
    def __init__(self, df, evalMode="all"):

        # index start at one and index zero is used for masking
        df.uid = df.uid.astype('category').cat.codes.values + 1
        df.iid = df.iid.astype('category').cat.codes.values + 1

        uNum = df.uid.max() + 1
        iNum = df.iid.max() + 1

        df = df.sort_values(["uid", "timestamp"], inplace=True)

        self.testRatings = {i[0]:i[1] for i in df.groupby("uid").tail(1)[["uid", "iid"]].values.tolist()}

        # for each user, remove last interaction from training set
        df = df.groupby("uid", as_index=False).apply(lambda x: x.iloc[:-1])

        mat = sp.dok_matrix((uNum, iNum), dtype=np.float32)
        seq = defaultdict(list)

        for u, i in df[["uid", "iid"]].values.tolist():
            mat[u, i] = 1.0
            seq[u].append(i)

        self.trainMatrix = mat
        self.trainSeq = seq
        self.df = df
        self.trainList = seq

        random.seed(2019)
        candidates = df.iid.tolist()

        self.testNegatives = defaultdict(list)
        for u in self.testRatings:
            gtItem = self.testRatings[u]
            if evalMode == "all":
                negs = set(range(iNum)) - set(self.trainSeq[u])
                if gtItem in negs:
                    negs.remove(gtItem)
                negs.remove(0) # remove masking venue, i.e. 0
            else:
                for i in range(100):
                    r = random.choice(candidates)
                    while (u, r) in mat or self.testRatings[u][1] == r:
                        r = random.choice(candidates)
                    negs.append(r)
            self.testNegatives[u] = list(negs)

        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape


class HeDataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path, mode=0):
        '''
        Constructor
        '''
        if mode == 0:
            self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
            self.trainList = self.load_training_file_as_list(path + ".train.rating")
            self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
            self.testNegatives = self.load_negative_file(path + ".test.negative")

            # extract seq data
            names = ["uid", "iid", "rating", "timestamp"]
            self.df = pd.read_csv(path + ".train.rating", sep="\t", names=names)
            self.df.sort_values(["uid", "timestamp"], inplace=True)
            self.trainSeq = defaultdict(list)

            for u, i in self.df[["uid", "iid"]].values.tolist():
                self.trainSeq[u].append(i)


        elif mode == 1:
            self.trainMatrix = self.load_training_file_as_matrix(path + "Train")
            self.trainList = self.load_training_file_as_list(path + "Train")
            self.testRatings = self.load_rating_file_as_list(path + "Test")
            self.testNegatives = self.load_negative_file(path + "TestNegative")

        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        print("already load the trainMatrix...")
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                # if index<300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        print("already load the trainList...")
        return lists


class OriginalDataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
        self.trainList = self.load_training_file_as_list(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        # self.testNegatives = self.load_negative_file(path + ".test.negative")
        # assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

        # extract seq data
        names = ["uid", "iid", "rating", "timestamp"]
        self.df = pd.read_csv(path + ".train.rating", sep="\t", names=names)
        self.trainSeq = defaultdict(list)

        for u, i in self.df[["uid", "iid"]].values.tolist():
            self.trainSeq[u].append(i)

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        print("already load the trainMatrix...")
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                #if index<300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        print("already load the trainList...")
        return lists