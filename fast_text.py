import fasttext

model = fasttext.train_supervised("./JD_binary/tokenized_train.csv", dim=300, label="",
                                  pretrainedVectors="./pretrained/cc.zh.300.vec")

print(model.labels)
print(model.words)


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


print_results(*model.test("./JD_binary/tokenized_test.csv"))
