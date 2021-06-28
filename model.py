import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        channel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            channel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class ModifiedTextCNN(nn.Module):
    def __init__(self, args):
        super(ModifiedTextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        channel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes
        char_filter_sizes = args.char_filter_sizes

        embedding_dimension = args.embedding_dim

        self.word_embedding = nn.Embedding.from_pretrained(args.word_vectors, freeze=not args.non_static)
        self.char_embedding = nn.Embedding.from_pretrained(args.char_vectors, freeze=not args.non_static)

        self.word_embed_dropout = nn.Dropout(args.dropout)
        self.char_embed_dropout = nn.Dropout(args.dropout)

        self.word_convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes]
        )

        self.char_convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dimension)) for size in char_filter_sizes]
        )

        self.word_conv_dropout = nn.Dropout(args.dropout)
        self.char_conv_dropout = nn.Dropout(args.dropout)

        self.fc = nn.Linear(len(filter_sizes) * filter_num * 2, class_num)
        self.fc_dropout = nn.Dropout(args.dropout)

    def forward(self, word, char):
        word = self.word_embedding(word)  # (batch_size, seq_len, embed_size)
        word = self.word_embed_dropout(word)
        word = word.unsqueeze(1)  # (batch_size, conv_in_channels=1, seq_len, embed_size)

        word = [F.relu(conv(word)).squeeze(3) for conv in self.word_convs]
        # (batch_size, conv_out_channels, seq_len, 1)  * num_filters (before squeeze)
        # (batch_size, conv_out_channels, seq_len)  * num_filters (after squeeze)

        word = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in word]
        # (batch_size, conv_out_channels, 1) * num_filters (before squeeze)
        # (batch_size, conv_out_channels) * num_filters (after squeeze)

        word = torch.cat(word, 1)  # (batch_size, conv_out_channels * num_filters)
        word = self.word_conv_dropout(word)

        char = self.char_embedding(char)
        char = self.char_embed_dropout(char)
        char = char.unsqueeze(1)
        char = [F.relu(conv(char)).squeeze(3) for conv in self.char_convs]
        char = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in char]
        char = torch.cat(char, 1)
        char = self.char_conv_dropout(char)

        out = torch.cat([word, char], 1)
        logits = self.fc(out)
        logits = self.fc_dropout(logits)

        return logits
