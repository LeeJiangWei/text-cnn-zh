import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchtext.data as data
from torchtext.vocab import Vectors

import dataset
import model

parser = argparse.ArgumentParser(description='TextCNN text classifier')

# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=2, help='number of epochs for train [default: 2]')
parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=4000,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=16000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')
parser.add_argument('-char-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

# channels
parser.add_argument('-static', type=bool, default=True, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=True,
                    help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')

# pretrained word vectors
parser.add_argument('-word-pretrained-name', type=str, default='cc.zh.300.vec',
                    help='filename of pre-trained word vectors')
parser.add_argument('-char-pretrained-name', type=str, default='fasttext_char300.vec',
                    help='filename of pre-trained char vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: 0]')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-vocab', type=bool, default=True, help='whether to load vocab cache')
parser.add_argument('-name', type=str, default="modified_all-zh", help='name of the model')
args = parser.parse_args()


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path, max_vectors=200000)
    return vectors


def load_dataset(text_field, char_field, label_field, args, **kwargs):
    print('Loading data...')
    train_dataset, dev_dataset, test_dataset = dataset.get_jd_dataset2(text_field, char_field, label_field)

    if not args.vocab:
        if args.static and args.word_pretrained_name and args.char_pretrained_name and args.pretrained_path:
            print("Loading pretrained vectors...")
            word_vectors = load_word_vectors(args.word_pretrained_name, args.pretrained_path)
            char_vectors = load_word_vectors(args.char_pretrained_name, args.pretrained_path)
            print("Building vocabulary...")
            text_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=word_vectors)
            char_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=char_vectors)
        else:
            print("Building vocabulary...")
            text_field.build_vocab(train_dataset, dev_dataset, test_dataset)
            char_field.build_vocab(train_dataset, dev_dataset, test_dataset)

        label_field.build_vocab(train_dataset, dev_dataset)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_dataset, dev_dataset, test_dataset),
                                                                 batch_size=args.batch_size,
                                                                 sort_key=lambda x: len(x.text),
                                                                 **kwargs)  #
    return train_iter, dev_iter, test_iter


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            text, char, target = batch.text, batch.char, batch.label
            if args.cuda:
                text, char, target = text.cuda(), char.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(text, char)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        # save(model, args.name, 'best', steps)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        text, char, target = batch.text, batch.char, batch.label
        if args.cuda:
            text, char, target = text.cuda(), char.cuda(), target.cuda()
        logits = model(text, char)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def save(model, name, save_prefix, steps):
    save_dir = os.path.join("./snapshot", name)
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


text_field = data.Field(fix_length=50, batch_first=True)
char_field = data.Field(fix_length=100, batch_first=True)
label_field = data.Field(sequential=False, is_target=True)

vocab_path = os.path.join("./vocab", args.name)
os.makedirs(vocab_path, exist_ok=True)
if args.vocab:
    print("Loading vocab...")
    text_field.vocab = torch.load(os.path.join(vocab_path, "text_vocab.pth"))
    char_field.vocab = torch.load(os.path.join(vocab_path, "char_vocab.pth"))
    label_field.vocab = torch.load(os.path.join(vocab_path, "label_vocab.pth"))
    train_iter, dev_iter, test_iter = load_dataset(text_field, char_field, label_field, args,
                                                   device=-1, repeat=False, shuffle=True)
else:
    train_iter, dev_iter, test_iter = load_dataset(text_field, char_field, label_field, args,
                                                   device=-1, repeat=False, shuffle=True)
    torch.save(text_field.vocab, os.path.join(vocab_path, "text_vocab.pth"))
    torch.save(char_field.vocab, os.path.join(vocab_path, "char_vocab.pth"))
    torch.save(label_field.vocab, os.path.join(vocab_path, "label_vocab.pth"))

if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.word_vectors = text_field.vocab.vectors
    args.char_vectors = char_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
args.char_filter_sizes = [int(size) for size in args.char_filter_sizes.split(',')]

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors', 'char_vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

text_cnn = model.ModifiedTextCNN(args)
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
    train(train_iter, dev_iter, text_cnn, args)
except KeyboardInterrupt:
    print('Exiting from training early')
finally:
    print("\n***** Test Accuracy *****")
    eval(test_iter, text_cnn, args)
