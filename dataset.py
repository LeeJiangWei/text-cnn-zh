import jieba
from torchtext import data


def word_cut(text):
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut

    train, dev, test = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='dev.tsv', test='test.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )

    return train, dev, test


def get_jd_dataset(text_field, label_field):
    # text_field.tokenize = word_cut

    train, dev, test = data.TabularDataset.splits(
        path="./JD_binary",
        format="csv",
        skip_header=False,
        train="tokenized_dev.csv",
        validation="tokenized_dev.csv",
        test="tokenized_dev.csv",
        fields=[
            ("label", label_field),
            ("text", text_field)
        ]
    )

    return train, dev, test


def get_jd_test(text_field, label_field):
    text_field.tokenize = word_cut

    test = data.TabularDataset.splits(
        path="./JD_binary",
        format="csv",
        skip_header=False,
        test="test.csv",
        fields=[
            ("label", label_field),
            ("text", text_field)
        ]
    )

    return test
