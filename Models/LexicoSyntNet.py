'''The implemntations of Siamese networks and the related functions are used from the following reference:
https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
'''


from __future__ import absolute_import
from __future__ import print_function
import pickle
import keras.optimizers
import logging, io, os, sys, time
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, Embedding,Activation
from keras.initializers import RandomUniform
from keras.callbacks import ModelCheckpoint
from DataLoader import *
from keras import backend as K
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.models import load_model
import multiprocessing

np.random.seed(725)

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../SentEval/data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)

from SentEval import senteval

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LexicoSyntNet:

    def __init__(self, embed_dim, rep_dim, vocab_size, num_struct_labels, lex_input_len, synt_input_len, base_network):
        self.embed_dim= embed_dim
        self.rep_dim = rep_dim
        self.vocab_size = vocab_size
        self.num_struct_labels = num_struct_labels
        self.lex_input_len = lex_input_len
        self.synt_input_len = synt_input_len
        self.base_network = base_network
        self. lexical_subnet, self.syntactic_subnet, self.lexicosyntnet = self.create_network()
        self.lex_embeddings = dict()

    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def compute_accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)

    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    def create_base_network_self_attention(self, vocab_size, embed_dim, rep_dim, input_length):
        '''Base network to be shared (eq. to feature extraction).'''

        #ToDo:Word embeddings need to be initialied fro unifrom
        sentence_in = Input(shape=(input_length,))
        word_embeddings = Embedding(input_dim=vocab_size,
                                    output_dim=embed_dim,
                                    input_length=input_length,
                                    trainable=True,
                                    embeddings_initializer=RandomUniform( minval=-0.1, maxval=0.1, seed=1000))(sentence_in)

        sentence_encoder = Bidirectional(LSTM(units=256, activation='relu', return_sequences=True))(word_embeddings)
        sentence_encoder = SeqSelfAttention(attention_activation='sigmoid',
                                            kernel_regularizer=keras.regularizers.l2(1e-4),
                                            bias_regularizer=keras.regularizers.l1(1e-4))(sentence_encoder)
        sentence_encoder = keras.layers.GlobalMaxPooling1D()(sentence_encoder)
        sentence_encoder = Dense(rep_dim)(sentence_encoder)
        sentence_encoder = Activation('tanh')(sentence_encoder)

        model = Model(inputs=sentence_in, outputs=sentence_encoder)
        model.summary()
        return model

    def create_network(self):

        lexical_subnetwork = self.create_base_network_self_attention(vocab_size=self.vocab_size, embed_dim=self.embed_dim,
                                                                     rep_dim=self.rep_dim, input_length=self.lex_input_len)
        syntactic_subnetwork = self.create_base_network_self_attention(vocab_size=self.num_struct_labels,
                                                                       embed_dim=self.embed_dim, rep_dim=self.rep_dim,
                                                                       input_length=self.synt_input_len)

        input_a = Input(shape=(self.lex_input_len,))
        input_b = Input(shape=(self.synt_input_len,))
        processed_a = lexical_subnetwork(input_a)
        processed_b = syntactic_subnetwork(input_b)
        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)
        # model_lexical = Model(inputs=input_a, outputs=processed_a)
        model.summary()
        return lexical_subnetwork, processed_b, model

    def train_network(self, data, epochs=50, batch_size=400):

        optimizer = keras.optimizers.nadam(0.0001)
        checkpoint = ModelCheckpoint(filepath="%s_flatt_40_80-epoch{epoch:02d}.hdf5" %(self.base_network), verbose=1, save_weights_only=True)
        callbacks_list = [checkpoint]
        steps_per_epoch = int(2*data.n_sentences//batch_size)
        data_generator = data.data_generator(batch_size=batch_size)

        self.lexicosyntnet.compile(loss=self.contrastive_loss, optimizer=optimizer, metrics=[self.accuracy])
        training = self.lexicosyntnet.fit_generator(generator= data_generator,
                                                    steps_per_epoch=steps_per_epoch,
                                                    epochs=epochs,
                                                    verbose=1,
                                                    callbacks=callbacks_list,
                                                    validation_data=[data.val_x, data.val_y],
                                                    validation_steps=None,
                                                    class_weight=None,
                                                    max_queue_size=1000,
                                                    workers=40,
                                                    use_multiprocessing=True,
                                                    shuffle=True,
                                                    initial_epoch=0)

        print('pickling...')
        with open("training.history_alldata_%s_%sepochs_%sbatch.p" %(self.base_network,epochs, batch_size), "wb") as fid:
            pickle.dump(training.history, fid, -1)


        #############-------------------Writing word embeddings to file---------------------############################
        self.lex_embeddings = self.lexical_subnet.layers[1].get_weights()[0]
        word_index = data.tokenizer.word_index
        assert self.lex_embeddings.shape[0]==data.vocab_size , "wrong layer"

        with open("word_vecs_alldata_%s_%sepochs_%sbatch" %(self.base_network, epochs, batch_size), "w") as f:
            line =""
            for word, index in word_index.items():
                line = str(word)+ " "+ " ".join(["{:.4f}".format(x) for x in self.lex_embeddings[index]])
                f.write(line)
                f.write("\n")

        f.close()

        #####################-------------------Writing syntactic embeddings to file---------------------###############
        synt_embeddings = self.syntactic_subnet.layers[1].get_weights()[0]
        assert synt_embeddings.shape[0]==len(data.POS_TAGS ), "wrong layer"

        with open("structural_labels_vecs_alldata_%s_%sepochs_%sbatch" %(self.base_network, epochs, batch_size), "w") as f:
            line =""
            for index, tag in enumerate(data.POS_TAGS):
                line = str(tag)+ " "+ " ".join(["{:.4f}".format(x) for x in synt_embeddings[index]])
                f.write(line)
                f.write("\n")

        f.close()

    def load_network(self, model_file):
        self.lexical_subnet, self.syntactic_subnet, self.lexicosyntnet = self.create_network()
        self.lexicosyntnet.load_weights(model_file)
        print("number of lyers: ",len(self.lexicosyntnet.layers))
        self.lex_embeddings = self.lexicosyntnet.layers[2].get_weights()[0]
        print("embedding shape: ", self.lex_embeddings.shape)

    def evaluate_lexicalsubnet(self, data):

        # lex_rep_func = K.function(inputs=self.lexicosyntnet.input[0], outputs=self.lexical_subnet)
        lex_rep_func = self.lexical_subnet
        embed_dim = self.embed_dim
        lex_embeddings = self.lex_embeddings

        def prepare(params, samples):
            params.word2id = data.tokenizer.word_index
            embeddings = lex_embeddings
            params.word_vec = {w:embeddings[idx] for w, idx in data.tokenizer.word_index.items()}
            # print("word_vec", len(params.word_vec))
            params.wvec_dim = embed_dim
            return

        def batcher(params, batch):
            batch = [sent if sent != [] else ['.'] for sent in batch]
            batch_input = []

            for index, sent in enumerate(batch):
                lex_in, synt_in = data.preprocess_input(sentence=sent, parse_tree=[], tokenizer=data.tokenizer)
                lex_in = np.array(lex_in, dtype='float32', ndmin=2).reshape((1, self.lex_input_len))
                # synt_in = np.array(synt_in, dtype='float32', ndmin=2).reshape((1, self.lex_input_len))
                # lexvec = lex_rep_func.predict([lex_in, synt_in])
                lexvec = lex_rep_func.predict(lex_in)
                # print(lexvec, lexvec.shape)
                batch_input.append(lexvec)

            batch_input = np.vstack(batch_input)
            return batch_input


        # Set params for SentEval
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}

        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber',
                         'ObjNumber', 'OddManOut', 'CoordinationInversion']  #
        results = se.eval(probing_tasks)
        print(results)
        with open("evaluation_lexicalsubnet_results_alldata.p", "wb") as fid:
            pickle.dump(results, fid, -1)

    def evaluate_bow(self, data):

        embed_dim = self.embed_dim
        embeddings = self.lex_embeddings

        def create_dictionary(sentences, threshold=0):
            words = {}
            for s in sentences:
                for word in s:
                    word = word.lower()
                    words[word] = words.get(word, 0) + 1

            if threshold > 0:
                newwords = {}
                for word in words:
                    if words[word] >= threshold:
                        newwords[word] = words[word]
                words = newwords
            words['<s>'] = 1e9 + 4
            words['</s>'] = 1e9 + 3
            words['<p>'] = 1e9 + 2

            sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
            id2word = []
            word2id = {}
            for i, (w, _) in enumerate(sorted_words):
                id2word.append(w)
                word2id[w] = i

            return id2word, word2id

        def get_wordvec(word2id):
            word_vec = {}

            for word, index in data.tokenizer.word_index.items():
                if word in word2id.keys():
                    word_vec[word] = embeddings[index]
                if word=='unk':
                    word_vec[word] = embeddings[index]

            logging.info('Found {0} words with word vectors, out of \
                {1} words'.format(len(word_vec), len(word2id)))
            # print(word_vec)
            return word_vec


        def prepare(params, samples):
            _, params.word2id = create_dictionary(samples)
            params.word_vec = get_wordvec(params.word2id)
            params.wvec_dim = embed_dim
            return

        def batcher(params, batch):
            batch = [sent if sent != [] else ['.'] for sent in batch]
            embeddings = []

            for sent in batch:
                sentvec = []
                for word in sent:
                    word = word.lower()
                    if word in params.word_vec:
                        sentvec.append(params.word_vec[word])
                    else:
                        sentvec.append(params.word_vec['unk'])
                if not sentvec:
                    vec = np.zeros(params.wvec_dim)
                    sentvec.append(vec)
                sentvec = np.mean(sentvec, 0)
                embeddings.append(sentvec)

            embeddings = np.vstack(embeddings)
            return embeddings

        #seneval params
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}

        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber',
                         'ObjNumber', 'OddManOut', 'CoordinationInversion']  #
        results = se.eval(probing_tasks)
        print(results)
        with open("evaluation_bow_results_alldata.p", "wb") as fid:
            pickle.dump(results, fid, -1)

    def evaluate_bow_fasttext(self, data):

        embed_dim = self.embed_dim+300
        embeddings = self.lex_embeddings

        def create_dictionary(sentences, threshold=0):
            words = {}
            for s in sentences:
                for word in s:
                    word = word.lower()
                    words[word] = words.get(word, 0) + 1

            if threshold > 0:
                newwords = {}
                for word in words:
                    if words[word] >= threshold:
                        newwords[word] = words[word]
                words = newwords
            words['<s>'] = 1e9 + 4
            words['</s>'] = 1e9 + 3
            words['<p>'] = 1e9 + 2

            sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
            id2word = []
            word2id = {}
            for i, (w, _) in enumerate(sorted_words):
                id2word.append(w)
                word2id[w] = i

            return id2word, word2id

        def get_wordvec(word2id):
            word_vec = {}
            PATH_TO_Embed = '../SentEval/examples/fasttext/crawl-300d-2M.vec'

            with io.open(PATH_TO_Embed, 'r', encoding='utf-8') as f:
                # if word2vec or fasttext file : skip first line "next(f)"
                for line in f:
                    word, vec = line.split(' ', 1)
                    if word in word2id.keys():
                        word_vec[word] = np.fromstring(vec, sep=' ')
                    if word == 'unk':
                        word_vec[word] = np.fromstring(vec, sep=' ')

            for word, vec in word_vec.items():
                if word in data.tokenizer.word_index.keys():
                    word_vec[word] = np.concatenate([vec,embeddings[data.tokenizer.word_index[word]]])
                else:
                    word_vec[word] = np.concatenate([vec, embeddings[data.tokenizer.word_index['unk']]])

            logging.info('Found {0} words with word vectors, out of \
                {1} words'.format(len(word_vec), len(word2id)))
            # print(word_vec)
            return word_vec


        def prepare(params, samples):
            _, params.word2id = create_dictionary(samples)
            params.word_vec = get_wordvec(params.word2id)
            params.wvec_dim = embed_dim
            return

        def batcher(params, batch):
            batch = [sent if sent != [] else ['.'] for sent in batch]
            embeddings = []

            for sent in batch:
                sentvec = []
                for word in sent:
                    word = word.lower()
                    if word in params.word_vec:
                        sentvec.append(params.word_vec[word])
                    else:
                        sentvec.append(params.word_vec['unk'])

                if not sentvec:
                    vec = np.zeros(params.wvec_dim)
                    sentvec.append(vec)
                sentvec = np.mean(sentvec, 0)
                embeddings.append(sentvec)

            embeddings = np.vstack(embeddings)
            return embeddings

        # Set params for SentEval
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}

        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber',
                         'ObjNumber', 'OddManOut', 'CoordinationInversion']  #
        results = se.eval(probing_tasks)
        print(results)
        with open("evaluation_bow+fasttext_results_alldata.p", "wb") as fid:
            pickle.dump(results, fid, -1)

    def evaluate_bow_glove(self, data):

        embed_dim = self.embed_dim + 300
        embeddings = self.lex_embeddings

        def create_dictionary(sentences, threshold=0):
            words = {}
            for s in sentences:
                for word in s:
                    word = word.lower()
                    words[word] = words.get(word, 0) + 1

            if threshold > 0:
                newwords = {}
                for word in words:
                    if words[word] >= threshold:
                        newwords[word] = words[word]
                words = newwords
            words['<s>'] = 1e9 + 4
            words['</s>'] = 1e9 + 3
            words['<p>'] = 1e9 + 2

            sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
            id2word = []
            word2id = {}
            for i, (w, _) in enumerate(sorted_words):
                id2word.append(w)
                word2id[w] = i

            return id2word, word2id

        def get_wordvec(word2id):
            word_vec = {}
            PATH_TO_Embed = '../SentEval/examples/glove/glove.840B.300d.txt'

            with io.open(PATH_TO_Embed, 'r', encoding='utf-8') as f:
                # if word2vec or fasttext file : skip first line "next(f)"
                for line in f:
                    word, vec = line.split(' ', 1)
                    if word in word2id.keys():
                        word_vec[word] = np.fromstring(vec, sep=' ')
                    if word == 'unk':
                        word_vec[word] = np.fromstring(vec, sep=' ')

            for word, vec in word_vec.items():
                if word in data.tokenizer.word_index.keys():
                    word_vec[word] = np.concatenate([vec, embeddings[data.tokenizer.word_index[word]]])
                else:
                    word_vec[word] = np.concatenate([vec, embeddings[data.tokenizer.word_index['unk']]])

            logging.info('Found {0} words with word vectors, out of \
                {1} words'.format(len(word_vec), len(word2id)))
            # print(word_vec)
            return word_vec

        def prepare(params, samples):
            _, params.word2id = create_dictionary(samples)
            params.word_vec = get_wordvec(params.word2id)
            params.wvec_dim = embed_dim
            return

        def batcher(params, batch):
            batch = [sent if sent != [] else ['.'] for sent in batch]
            embeddings = []

            for sent in batch:
                sentvec = []
                for word in sent:
                    word = word.lower()
                    if word in params.word_vec:
                        sentvec.append(params.word_vec[word])
                    else:
                        sentvec.append(params.word_vec['unk'])

                if not sentvec:
                    vec = np.zeros(params.wvec_dim)
                    sentvec.append(vec)
                sentvec = np.mean(sentvec, 0)
                embeddings.append(sentvec)

            embeddings = np.vstack(embeddings)
            return embeddings

        # Set params for SentEval
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}

        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber',
                         'ObjNumber', 'OddManOut', 'CoordinationInversion']  #
        results = se.eval(probing_tasks)
        print(results)
        with open("evaluation_bow+fasttext_results_alldata.p", "wb") as fid:
            pickle.dump(results, fid, -1)

    # def visualize_attention(self):


if __name__=="__main__":

    data = pickle.load(open("lambda_data_parsetrees_fasttext_len_40.p", "rb"))
    network = LexicoSyntNet(embed_dim=100,rep_dim=100, vocab_size=data.vocab_size, num_struct_labels=len(data.POS_TAGS), lex_input_len=40, synt_input_len=40, base_network="self_att_max")
    # network.load_network(model_file="self_att_max-epoch45.hdf5")
    network.train_network(data=data, epochs=50, batch_size=400)
    print(time.strftime('%l:%M%p'), "------------------------------BOW evaluation----------------------------------")
    network.evaluate_bow(data)
    print(time.strftime('%l:%M%p'), "-----------------------------BOW+glove evaluation----------------------------")
    network.evaluate_bow_glove(data)
    print(time.strftime('%l:%M%p'), "--------------------------------lexicalsubnetwork evaluation-----------------------------")
    network.evaluate_lexicalsubnet(data)


