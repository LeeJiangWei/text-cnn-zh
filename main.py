import argparse
import os

import torch
import torchtext.data as data
from torchtext.vocab import Vectors

import dataset
import model
import train

parser = argparse.ArgumentParser(description='TextCNN text classifier')

# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=2, help='number of epochs for train [default: 2]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=500,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=2000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

# channels
parser.add_argument('-static', type=bool, default=True, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=True,
                    help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')

# pretrained word vectors
parser.add_argument('-pretrained-name', type=str, default='sgns.word-character.vec',
                    help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: 0]')

# option
parser.add_argument('-snapshot', type=str, default=None,
                    help='filename of model snapshot [default: None]')
parser.add_argument('-name', type=str, default="test", help='name of the model')
args = parser.parse_args()


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    print('Loading data...')
    train_dataset, dev_dataset, test_dataset = dataset.get_jd_dataset(text_field, label_field)

    if args.static and args.pretrained_name and args.pretrained_path:
        print("Loading pretrained vectors..")
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        print("Building vocabulary...")
        text_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=vectors)
    else:
        print("Building vocabulary...")
        text_field.build_vocab(train_dataset, dev_dataset, test_dataset)

    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_dataset, dev_dataset, test_dataset),
                                                                 batch_size=args.batch_size,
                                                                 sort_key=lambda x: len(x.text),
                                                                 **kwargs)
    return train_iter, dev_iter, test_iter


def load_testset(text_field, label_field, args, **kwargs):
    print('Loading testing set...')
    test_dataset = dataset.get_jd_test(text_field, label_field)

    test_iter, = data.BucketIterator.splits(test_dataset,
                                            batch_size=args.batch_size,
                                            sort_key=lambda x: len(x.text),
                                            **kwargs)
    return test_iter


text_field = data.Field(fix_length=50, batch_first=True)
label_field = data.Field(sequential=False, is_target=True)

vocab_path = os.path.join("./vocab", args.name)
os.makedirs(vocab_path, exist_ok=True)
if args.snapshot:
    print("Loading vocab...")
    text_field.vocab = torch.load(os.path.join(vocab_path, "text_vocab.pth"))
    label_field.vocab = torch.load(os.path.join(vocab_path, "label_vocab.pth"))
    test_iter = load_testset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
else:
    train_iter, dev_iter, test_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
    torch.save(text_field.vocab, os.path.join(vocab_path, "text_vocab.pth"))
    torch.save(label_field.vocab, os.path.join(vocab_path, "label_vocab.pth"))

args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

text_cnn = model.TextCNN(args)
if args.snapshot:
    model_path = os.path.join("snapshot", args.name, args.snapshot)
    print('\nLoading model from {}...\n'.format(model_path))
    text_cnn.load_state_dict(torch.load(model_path))

for name, param in text_cnn.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()

try:
    if not args.snapshot:
        train.train(train_iter, dev_iter, text_cnn, args)
except KeyboardInterrupt:
    print('Exiting from training early')
finally:
    print("\n***** Test Accuracy *****")
    train.eval(test_iter, text_cnn, args)
