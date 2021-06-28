import fasttext


def text_cls():
    model = fasttext.train_supervised("./JD_binary/fasttext_dev.csv", dim=300)

    print(model.labels)
    print(model.words)

    def print_results(N, p, r):
        print("N\t" + str(N))
        print("P@{}\t{:.7f}".format(1, p))
        print("R@{}\t{:.7f}".format(1, r))

    print_results(*model.test("./JD_binary/fasttext_test.csv"))


def train_vector():
    model = fasttext.train_unsupervised("./JD_binary/fasttext_uns.csv", dim=300, minn=1, maxn=2)
    model.save_model("pretrained/fasttext_char300.bin")


def bin_to_vec():
    # original BIN model loading
    f = fasttext.load_model("pretrained/fasttext_char300.bin")

    # get all words from model
    words = f.get_words()

    with open("pretrained/fasttext_char300.vec", 'w', encoding="utf-8") as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass


if __name__ == '__main__':
    bin_to_vec()
