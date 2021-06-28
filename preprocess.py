import pandas as pd
import jieba


def preprocess():
    df = pd.read_csv("./JD_binary/binary_test.csv", header=None)

    df = df[df[2].str.contains(r'[\u4e00-\u9fa5]+')]  # delete rows that do not contain Chinese
    df[2] = df[2].replace(r'[^\u4e00-\u9fa50-9]', ' ', regex=True)  # rm non-zh chars
    df[2] = df[2].str.strip()
    df[0] = df[0].map(lambda x: 0 if x == 2 else 1)
    del df[1]

    df.info()
    df.to_csv("./JD_binary/test.csv", header=None, index=None)

    # dev = df.sample(n=1000)
    # dev.to_csv("./JD_binary/dev.csv", header=None, index=None)


def duplicate(i, o):
    df = pd.read_csv(i, header=None)
    df[2] = df[1]
    df[2] = df[2].replace(r'[^\u4e00-\u9fa5]', '', regex=True)
    df.to_csv(o, header=None, index=None)


def del_non_zh(i, o):
    df = pd.read_csv(i, header=None)
    df[1] = df[1].replace(r'[^\u4e00-\u9fa5]', '', regex=True)
    df.to_csv(o, header=None, index=None)


def tokenize_dataset(i, o):
    df = pd.read_csv(i, header=None)
    df[1] = df[1].map(lambda x: " ".join(list(jieba.cut(x))))
    df.to_csv(o, header=None, index=None)


def format_fasttext():
    df = pd.read_csv("./JD_binary/tokenized_test2.csv", header=None)
    df[0] = df[0].map(lambda x: "__label__" + str(x))
    del df[2]
    df.to_csv("./JD_binary/fasttext_test.csv", header=None, index=None, sep=" ")


def format_fasttext_unsupervised():
    train = pd.read_csv("./JD_binary/tokenized_train2.csv", header=None)
    dev = pd.read_csv("./JD_binary/tokenized_dev2.csv", header=None)
    test = pd.read_csv("./JD_binary/tokenized_test2.csv", header=None)

    df = pd.concat([train, dev, test])
    df[2] = df[2].map(lambda x: " ".join([i for i in x]))
    del df[1]
    del df[0]

    df.to_csv("./JD_binary/fasttext_uns.csv", header=None, index=None)


def train_dev_split():
    train = pd.read_csv("./JD_binary/tokenized_train2.csv", header=None)
    train.info()
    print(train.iloc[0].unique())

    dev = train.sample(frac=0.1)
    train = train.drop(dev.index)

    train.info()
    dev.info()

    dev.to_csv("./JD_binary/tokenized_dev2.csv", header=None, index=None)
    train.to_csv("./JD_binary/tokenized_train2.csv", header=None, index=None)


if __name__ == '__main__':
    input_path = "./JD_binary/tokenized_test2.csv"
    output_path = "./JD_binary/tokenized_test2.csv"
    format_fasttext_unsupervised()
