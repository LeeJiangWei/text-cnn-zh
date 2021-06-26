import re
import pandas as pd
import jieba


def preprocess():
    df = pd.read_csv("./JD_binary/binary_test.csv", header=None)

    df = df[df[2].str.contains(r'[\u4e00-\u9fa5]+')]
    df[2] = df[2].replace(r'[^\u4e00-\u9fa50-9]', ' ', regex=True)
    df[2] = df[2].str.strip()
    df[0] = df[0].map(lambda x: 0 if x == 2 else 1)
    del df[1]

    df.info()
    df.to_csv("./JD_binary/test.csv", header=None, index=None)

    # dev = df.sample(n=1000)
    # dev.to_csv("./JD_binary/dev.csv", header=None, index=None)


def duplicate():
    df = pd.read_csv("./JD_binary/dev.csv", header=None)
    df[2] = df[1]
    df.to_csv("./JD_binary/dev2.csv", header=None, index=None)


def tokenize_dataset():
    df = pd.read_csv("./JD_binary/dev.csv", header=None)
    df[1] = df[1].map(lambda x: " ".join(list(jieba.cut(x))))
    df.to_csv("./JD_binary/dev_tokenized.csv", header=None, index=None)


if __name__ == '__main__':
    tokenize_dataset()
