""" Data loader class using multithreading"""


import nltk
from nltk.tokenize import word_tokenize as word_tokenize
import random
import time, os, pickle, re
from keras.preprocessing.text import Tokenizer
from nltk.parse import CoreNLPParser
import numpy as np
np.random.seed(725)
import threading


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(iter(self.it))


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g



class DataLoader:
    """ Class constructor """

    def __init__(self, file_path, lex_len, synt_len, embedding, parse=True):

        self.val_size = 1000
        self.lex_len = lex_len
        self.synt_len = synt_len

        print(time.strftime('%l:%M%p'), "File path: ", file_path)
        print(time.strftime('%l:%M%p'), "Sentence length: ", self.lex_len, self.synt_len)

        """ The list of all possible pos tags from NLTK Parser """
        self.POS_TAGS = ['None','S', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP',
                         'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                         'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', ',', ':', ';', '...', '.', '!', '?', '$', '(', ')',
                         '``', '\'\'', 'unk']

        self.pos_to_index = {pos: index for index, pos in enumerate(self.POS_TAGS)}
        self.index_to_pos = {index: pos for index, pos in enumerate(self.POS_TAGS)}

        """ The list of all sentences + the tokeniser which is fitted on the data"""
        self.train_lex, self.train_synt, self.val_lex_positive, self.val_synt_positive, self.tokenizer = self.create_data(file_path, parse)
        self.val_x, self.val_y = self.create_val_data()



        self.vocab_size = len(self.tokenizer.word_index)+2
        # self.vocab_size= self.tokenizer.num_words+1
        print(time.strftime('%l:%M%p'), "Vocabulary Size:", self.vocab_size)

        self.n_sentences = len(self.train_lex)
        print(time.strftime('%l:%M%p'), "Number of sentences in the file:", self.n_sentences)

        self.embedding_weights = self.get_pretrained_embeddings(type=embedding)
        if self.embedding_weights is not None:
            print(time.strftime('%l:%M%p'), "Embedding type: ", embedding)
        else:
            print(time.strftime('%l:%M%p'), "No pretrained embeddings")

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!#$%&./:;?@`']", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def create_data(self, file_path, parse):
        tag_set = set(self.POS_TAGS)
        sentences = []
        parse_trees = []
        parser = CoreNLPParser(url='http://localhost:9000')

        def dfs(tree):
            if type(tree) is str:
                return

            tag =tree.label()
            if tag != 'ROOT':
                linearized_parse_tree.append(tag)
                tag_set.add(tag)

            for child in tree:
                dfs(child)

        if parse:
            for dir in os.listdir(file_path):
                print(time.strftime('%l:%M%p'), dir)
                rand_file = np.random.choice(os.listdir(file_path + "/" + dir), size=int(len(os.listdir(file_path + "/" + dir))), replace=False)
                for file in rand_file:
                    # print(file)
                    sent_batch, counter = [], 0
                    for line in open(file_path + "/" + dir + "/" + file, encoding='utf-8'):
                        line = self.clean_str(line).lower()
                        if len(line)>1:
                            tokens = word_tokenize(line)
                            if len(tokens) > self.lex_len:
                                tokens = tokens[0:self.lex_len]
                            sent_batch.append(tokens)
                            counter += 1
                            if counter == 200:

                                try:
                                    parsed_trees = parser.parse_sents(sent_batch)
                                    for index, tree in enumerate(parsed_trees):
                                        linearized_parse_tree = []
                                        dfs(next(tree))
                                        parse_trees.append(linearized_parse_tree[:self.synt_len])
                                        sentences.append(sent_batch[index])

                                    # for sent in sent_batch:
                                    #     sentences.append(sent)
                                    #
                                    sent_batch, counter = [], 0
                                except:
                                    print("something went wrong in %s",file )
                                    sent_batch, counter = [], 0



        else:
            for dir in os.listdir(file_path):
                print(time.strftime('%l:%M%p'), dir)
                for file in os.listdir(file_path + "/" + dir):
                    for line in open(file_path + "/" + dir + "/" + file, encoding='utf-8'):
                        line = self.clean_str(line)
                        if len(line) > 1:
                            line = line.lower()
                            sent, tags = [], []
                            sentence_taged = nltk.pos_tag(word_tokenize(line))
                            for index, token in enumerate(sentence_taged):
                                sent.insert(index, token[0])
                                tags.insert(index, token[1])
                            sentences.append(sent)
                            parse_trees.append(tags)

        assert len(sentences) == len(parse_trees)

        for tag in tag_set:
            if tag not in self.POS_TAGS:
                index = len(self.POS_TAGS)
                self.POS_TAGS.append(tag)
                self.pos_to_index[tag] = index
                self.index_to_pos[index] = tag

        self.num_tags = len(self.POS_TAGS)
        print(time.strftime('%l:%M%p'), "Number of part of speech tags:", self.num_tags)

        print(time.strftime('%l:%M%p'), "Fitting keras tokenizer on the dataset")
        tokenizer = Tokenizer(lower=True, filters='\t\n', oov_token = 'unk')
        tokenizer.fit_on_texts(sentences)
        print("unk index:", tokenizer.word_index['unk'])

        train_lex, train_synt = [], []
        for index, sent in enumerate(sentences):
            # print(sent)
            lex_in, synt_in = self.preprocess_input(sentence=sent, parse_tree=parse_trees[index], tokenizer=tokenizer)
            train_lex.append(lex_in)
            train_synt.append(synt_in)
        assert len(train_lex)==len(train_synt)

        val_lex, val_synt = [], []
        val_indices = np.random.choice(len(train_lex), size=self.val_size, replace=False)
        for index in reversed(sorted(val_indices)):
            val_lex.append(train_lex.pop(index))
            val_synt.append(train_synt.pop(index))
        assert len(val_lex)==len(val_synt)
        print(time.strftime('%l:%M%p'), "number of train and validation instances:", len(train_lex),len(val_lex))

        return train_lex, train_synt, val_lex, val_synt, tokenizer

    def preprocess_input(self, sentence, parse_tree, tokenizer):

        synt_seq, lex_seq = [], []
        for index, token in enumerate(sentence):
            if token in tokenizer.word_index.keys():
                lex_seq.insert(index, tokenizer.word_index[token])
            else:
                lex_seq.insert(index, tokenizer.word_index['unk'])

        for index, tag in enumerate(parse_tree):
            if tag in self.POS_TAGS:
                synt_seq.insert(index, self.pos_to_index[tag])
            else:
                synt_seq.insert(index,  self.pos_to_index['unk'])

        synt_len, lex_len = len(synt_seq), len(lex_seq)

        if synt_len<self.sent_len:
            for index in range(synt_len, self.sent_len):
                synt_seq.append(0)
        else:
            synt_seq = synt_seq[:self.sent_len]

        if lex_len<self.sent_len:
            for index in range(lex_len, self.sent_len):
                lex_seq.append(0)
        else:
            lex_seq = lex_seq[:self.sent_len]

        # if len(synt_seq) != len(lex_seq):
        #     raise ValueError('The syntactic and lexical sequence length do not match')

        return lex_seq, synt_seq

    def get_pretrained_embeddings(self, type="fasttext", embed_dim=300):

        ''' Creating Embedding Matix'''
        print(time.strftime('%l:%M%p'), "Creating Embedding Matix")

        embeddings_index = dict()
        if type == "fasttext":
            PATH_TO_Embed = '../SentEval/examples/fasttext/crawl-300d-2M.vec'
            print("embedding: FastText")
        else:
            PATH_TO_Embed = '../SentEval/examples/glove/glove.840B.300d.txt'
            print("embedding: Glove")

        f = open(PATH_TO_Embed)
        for line in f:
            # Note: use split(' ') instead of split() if you get an error.
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        not_found, counter = [], 0
        word_index = self.tokenizer.word_index
        embedding_matrix = (np.random.random_sample((self.vocab_size, embed_dim)) - 0.5) / 0.5
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector[0:embed_dim]
                counter += 1
            else:
                not_found.append(word)

        with open("words_not_found.p", "wb") as fid:
            pickle.dump(not_found, fid, -1)

        print(time.strftime('%l:%M%p'), "%s number of words found:" % counter)
        return embedding_matrix

    @threadsafe_generator
    def data_generator(self, batch_size):
        # count=0
        while True:
            batch_indices = np.random.choice(self.n_sentences, size=batch_size//2, replace=False)
            batch_input_lex, batch_input_synt, batch_output, temp_lex, temp_synt = [], [], [], [], []
            for index in batch_indices:
                lex_in , synt_in = self.train_lex[index], self.train_synt[index]
                batch_input_lex.append(lex_in)
                batch_input_synt.append(synt_in)
                batch_output.append(1)

            for idx, item in enumerate(batch_input_lex):
                inc = random.randrange(0, len(batch_input_lex))
                temp_lex.append(batch_input_lex[idx])
                temp_synt.append(batch_input_synt[inc])
                batch_output.append(0)

            batch_input_lex.extend(temp_lex)
            batch_input_synt.extend(temp_synt)
            batch_x_lex = np.array(batch_input_lex, dtype="int64", ndmin=2)
            batch_x_synt = np.array(batch_input_synt, dtype="int64", ndmin=2)
            batch_y = np.array(batch_output, ndmin=2).transpose()

            # batch_x.reshape(batch_size, self.sent_len,2)
            # batch_y.reshape(batch_size,1)

            # print("\nbatch: ", count)
            # count+=1
            assert not np.any(np.isnan(batch_x_synt))
            assert not np.any(np.isnan(batch_x_lex))
            assert not np.any(np.isnan(batch_y))
            yield ([batch_x_lex, batch_x_synt], batch_y)

    def create_val_data(self):

        batch_input_lex, batch_input_synt,  batch_output, temp_lex, temp_synt = [], [], [], [], []
        for index in range(self.val_size):
            lex_in, synt_in = self.val_lex_positive[index], self.val_synt_positive[index]
            batch_input_lex.append(lex_in)
            batch_input_synt.append(synt_in)
            batch_output.append(1)

            # print(lex_in)
            # print(synt_in)
            print(len(lex_in),len(synt_in))

        for idx, item in enumerate(batch_input_lex):
            inc = random.randrange(0, len(batch_input_lex))
            temp_lex.append(batch_input_lex[idx])
            temp_synt.append(batch_input_synt[inc])
            batch_output.append(0)

        batch_input_lex.extend(temp_lex)
        batch_input_synt.extend(temp_synt)
        print(len(batch_input_lex))
        print(len(batch_input_synt))


        batch_x_lex = np.array(batch_input_lex, dtype="int64", ndmin=2)
        batch_x_synt = np.array(batch_input_synt, dtype="int64",ndmin=2)
        batch_y = np.array(batch_output, ndmin=2).transpose()

        # batch_x.reshape(batch_size, self.sent_len,2)
        # batch_y.reshape(batch_size,1)
        # print(batch_x.shape, batch_y.shape)

        assert not np.any(np.isnan(batch_x_synt))
        assert not np.any(np.isnan(batch_x_lex))
        assert not np.any(np.isnan(batch_y))

        return [batch_x_lex, batch_x_synt], batch_y




if __name__=="__main__":

    lex_len, synt_len = 40, 80
    embedding = "fasttext"
    file_path = "../Datasets/lambdadataset/train-novels/"
    # file_path = "./sub_data"

    dl = DataLoader(file_path=file_path, lex_len=lex_len, synt_len = synt_len, embedding=embedding, parse=True)

    print('pickling...')
    with open("lambda_data_parsetrees_%s_len_%s_%s.p" %(embedding,lex_len,synt_len), "wb") as fid:
        pickle.dump(dl, fid, -1)
