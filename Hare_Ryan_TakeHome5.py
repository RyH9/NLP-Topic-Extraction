import os
import re
from sklearn import datasets as skdata
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.manifold import TSNE
from tensorflow import keras
import numpy as np
import pandas as pd
import gensim
import nltk
import string
from matplotlib import pyplot as plt


class CBOWGenerator:
    def __init__(self, input_text, load_weights=False, training_epochs=5):
        print("Initializing CBOW model...")
        # Remove all return characters and convert to lowercase
        for idx, s in enumerate(input_text):
            input_text[idx] = re.sub('\r', '', s).lower()

        # Remove stop words
        stop_words = nltk.corpus.stopwords.words('english')
        for idx, s in enumerate(input_text):
            for word in stop_words:
                input_text[idx] = re.sub(r"\b%s\b" % word, '', input_text[idx])

        # Initialize Keras tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(input_text)
        self.word2id = self.tokenizer.word_index

        self.word2id['PAD'] = 0
        self.id2word = self.tokenizer.index_word
        self.wids = [[self.word2id[w] for w in keras.preprocessing.text.text_to_word_sequence(doc)]
                     for doc in input_text]

        self.vocab_size = len(self.word2id)
        self.embed_size = 200
        self.window_size = 2

        # Make CBOW neural network model
        self.init_cbow_model()

        # Train the network or load the weights
        if not load_weights:
            print("Training network...")
            self.train_cbow(n_epochs=training_epochs)

            self.model.save_weights("cbow_network_weights")
        else:
            print("Loading weights...")
            self.model.load_weights("cbow_network_weights")

        weights = self.model.get_weights()[0]
        weights = weights[1:]

        # compute pairwise distance matrix
        self.distance_matrix = euclidean_distances(weights)

    def generate_context_word_pairs(self):
        context_length = self.window_size*2
        for words in self.wids:
            sentence_length = len(words)
            for index, word in enumerate(words):
                context_words = []
                label_word = []
                start = index - self.window_size
                end = index + self.window_size + 1

                context_words.append([words[i]
                                      for i in range(start, end)
                                      if 0 <= i < sentence_length
                                      and i != index])
                label_word.append(word)

                x = keras.preprocessing.sequence.pad_sequences(context_words, maxlen=context_length)
                y = keras.utils.to_categorical(label_word, self.vocab_size)
                yield (x, y)

    def init_cbow_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size,
                                              input_length=self.window_size*2))
        self.model.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1), output_shape=(self.embed_size,)))
        self.model.add(keras.layers.Dense(self.vocab_size, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    def train_cbow(self, n_epochs):
        for epoch in range(n_epochs):
            loss = 0.
            i = 0
            for x, y in self.generate_context_word_pairs():
                i += 1

                loss += self.model.train_on_batch(x, y)
                if i % 1000 == 0:
                    print('Processed {} (context, word) pairs'.format(i), 'loss=', loss)

            print('Epoch:', epoch, '\tLoss:', loss)
            print()


class Word2VecGenerator:
    def __init__(self, input_text):
        print("Initializing Gensim Word2Vec model...")
        # Remove all return characters and convert to lowercase
        for idx, s in enumerate(input_text):
            input_text[idx] = re.sub('\r', '', s).lower()

        # tokenize sentences in corpus
        wpt = nltk.WordPunctTokenizer()
        self.tokenized_corpus = [wpt.tokenize(document) for document in input_text]

        # Set values for various parameters
        feature_size = 200    # Word vector dimensionality
        window_context = 25          # Context window size
        min_word_count = 2   # Minimum word count
        sample = 1e-3   # Downsample setting for frequent words

        self.model = gensim.models.word2vec.Word2Vec(self.tokenized_corpus, vector_size=feature_size,
                                                     window=window_context, min_count=min_word_count, sample=sample,
                                                     epochs=50)


class FastTextGenerator:
    def __init__(self, input_text):
        print("Initializing FastText model...")
        # Remove all return characters and convert to lowercase
        for idx, s in enumerate(input_text):
            input_text[idx] = re.sub('\r', '', s).lower()

        wpt = nltk.WordPunctTokenizer()
        self.tokenized_corpus = [wpt.tokenize(document) for document in input_text]

        # Set values for various parameters
        feature_size = 200    # Word vector dimensionality
        window_context = 25         # Context window size
        min_word_count = 2   # Minimum word count
        sample = 1e-3   # Downsample setting for frequent words

        self.model = gensim.models.fasttext.FastText(self.tokenized_corpus, vector_size=feature_size,
                                                     window=window_context, min_count=min_word_count, sample=sample,
                                                     sg=1, epochs=50)


def process_all_html(in_dir, out_dir):
    """
    Function to read all HTML files in a given directory and output all text file data
    """
    n = 0
    # Find all files
    for file in os.listdir(in_dir):
        # Check if HTML
        if file.endswith(".html"):
            # Process
            process_html_file(os.path.join(in_dir, file), n, out_dir)
            n += 1


def process_html_file(file_path, n, output_path):
    """
    Function to read in a given HTML file path and output a data txt file in format Data_n
    :param file_path: Path to the input file
    :param n: Number for the output file
    :param output_path: output_path: Path to the output directory
    """

    # Open HTML file and read text
    f = open(file_path, 'r', encoding='utf-16-le', errors='ignore')
    html_text = f.read()

    # Strip HTML markers
    stripped_text = re.sub('<[^>]*>', '', html_text)

    # Strip headers
    start = stripped_text.find("From")
    end = len(stripped_text)
    stripped_text = stripped_text[start:end]

    # Strip other HTML artifacts
    stripped_text = re.sub('&gt;', '', stripped_text)
    stripped_text = re.sub('&#128227;', '', stripped_text)
    stripped_text = re.sub('&lt;', '', stripped_text)
    stripped_text = re.sub('&nbsp;', ' ', stripped_text)
    stripped_text = re.sub('&amp;', ' and ', stripped_text)
    stripped_text = re.sub('&quot;', '"', stripped_text)

    # Remove punctuation
    stripped_text = stripped_text.translate(str.maketrans('', '', string.punctuation))

    # Encode to ASCII
    s = stripped_text.encode('ascii', errors='ignore').decode()

    # Seperate lines
    ss = s.split('\n')

    wpt = nltk.WordPunctTokenizer()
    clean_text = []
    for line in ss:
        # Remove all return characters and convert to lowercase
        clean_text.append(re.sub('\r', '', line).lower())

        # tokenize sentences in corpus
        tokenized = wpt.tokenize(clean_text[-1])

        # Remove stop words and lemmatize
        wnl = nltk.stem.wordnet.WordNetLemmatizer()
        stop_words = nltk.corpus.stopwords.words('english')

        clean_text[-1] = [token for token in tokenized if token not in stop_words]
        clean_text[-1] = [wnl.lemmatize(token) for token in clean_text[-1] if not token.isnumeric()]
        clean_text[-1] = ' '.join(clean_text[-1])

    # Create directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create output file with given n
    output_txt = os.path.join(output_path, "data_%s.txt" % n)

    # Open output file
    f2 = open(output_txt, 'w+')

    # Print each line to the file
    for line in clean_text:
        # Clean multi-spaces
        line = re.sub(' +', ' ', line)
        ll = len(line)

        # ignore empty lines
        if ll > 2:
            f2.write(line + '\n')

    # Close file
    f2.close()


class LDATopicModelling:
    def __init__(self, input_data, total_topics=5, write_to_file=False):
        input_text = np.array([s.decode('ascii') for s in input_data['data'] if len(s) > 2])

        # Remove all return characters and convert to lowercase
        for idx, s in enumerate(input_text):
            input_text[idx] = re.sub('\r', '', s).lower()

        wpt = nltk.WordPunctTokenizer()
        self.tokenized_corpus = [wpt.tokenize(document) for document in input_text]

        cv = CountVectorizer(min_df=5, max_df=0.95, ngram_range=(1, 2),
                             token_pattern=None, tokenizer=lambda doc: doc,
                             preprocessor=lambda doc: doc)
        cv_features = cv.fit_transform(self.tokenized_corpus)

        self.vocabulary = np.array(cv.get_feature_names())

        lda_model = LatentDirichletAllocation(n_components =total_topics, max_iter=2500, max_doc_update_iter=100,
                                              learning_method='online', batch_size=150, learning_offset=20.,
                                              random_state=42, n_jobs=-1)
        document_topics = lda_model.fit_transform(cv_features)

        self.topic_terms = lda_model.components_

        top_terms = 20

        topic_key_term_idxs = np.argsort(-np.absolute(self.topic_terms), axis=1)[:, :top_terms]
        topic_keyterms = self.vocabulary[topic_key_term_idxs]
        topics = [', '.join(topic) for topic in topic_keyterms]
        pd.set_option('display.max_colwidth', -1)
        topics_df = pd.DataFrame(topics,
                                 columns = ['Terms per Topic'],
                                 index=['Topic'+str(t) for t in range(1, total_topics+1)])

        if write_to_file:
            # Clear results.txt
            file = open(r'results.txt', "w")
            file.close()

            topics_df.to_csv(r'results.txt', columns=['Terms per Topic'],
                             index=['Topic'+str(t) for t in range(1, total_topics+1)], sep=' ', mode='a')

        pd.options.display.float_format = '{:,.3f}'.format
        dt_df = pd.DataFrame(document_topics,
                             columns=['T'+str(i) for i in range(1, total_topics+1)])

        pd.options.display.float_format = '{:,.5f}'.format
        pd.set_option('display.max_colwidth', 200)

        max_contrib_topics = dt_df.max(axis=0)
        dominant_topics = max_contrib_topics.index
        contrib_perc = max_contrib_topics.values
        document_numbers = [dt_df[dt_df[t] == max_contrib_topics.loc[t]].index[0]
                            for t in dominant_topics]
        documents = [input_data['filenames'][i].split('\\')[-1] for i in document_numbers]

        results_df = pd.DataFrame({'Dominant Topic': dominant_topics, 'Contribution %': contrib_perc,
                                   'Email Num': document_numbers, 'Topic': topics_df['Terms per Topic'],
                                   'Email Name': documents})

        if write_to_file:
            results_df.to_csv(r'results.txt',
                              columns=['Dominant Topic', 'Contribution %', 'Email Num', 'Topic', 'Email Name'],
                              index=['Topic'+str(t) for t in range(1, total_topics+1)], sep=' ', mode='a')

        email_names = []
        for _ in range(total_topics):
            email_names.append([])

        for idx, doc in enumerate(document_topics):
            email_topic = np.argmax(np.array(doc))
            email_names[email_topic].append(input_data['filenames'][idx].split('\\')[-1])

        doc_topics_df = pd.DataFrame({'Email Names': email_names})

        if write_to_file:
            doc_topics_df.to_csv(r'results.txt', columns=['Email Names'],
                                 index=['Topic'+str(t) for t in range(1, total_topics+1)], sep=' ', mode='a')


def plot_model(model, topics, use_gensim=True, name=""):
    # List of defining terms for the 5 topics
    search_terms = ['3d', 'address', 'analysis', 'article', 'biotechniques', 'cell',
                    'chem', 'company', 'content', 'could', 'culture', 'daily', 'data',
                    'detail', 'development', 'digital', 'disease', 'dna', 'drug',
                    'edition', 'email', 'event', 'find', 'future', 'guideline', 'idea',
                    'leader', 'learn', 'life', 'market', 'may', 'mouse', 'nature',
                    'need', 'new', 'news', 'newsletter', 'number', 'pharma', 'phd',
                    'please', 'product', 'protein', 'read', 'recent', 'register',
                    'report', 'research', 'right', 'science', 'scientist', 'service',
                    'study', 'subscriber', 'system', 'table', 'technology',
                    'unsubscribe', 'update', 'view', 'webinar'
                    ]
    if use_gensim:
        # If using a gensim model, find similar words
        similar_words = {search_term: [item[0] for item in model.wv.most_similar([search_term], topn=5)]
                         for search_term in search_terms}

        words = sum([[k] + v for k, v in similar_words.items()], [])
        wvs = model.wv[words]

        pca = PCA(n_components=2)
        np.set_printoptions(suppress=True)
        P = pca.fit_transform(wvs)
        labels = words

        plt.figure(figsize=(18, 10))

        point_colors = []
        colors = ['black', 'red', 'limegreen', 'blue', 'magenta']
        for word in words:
            if word in topics.vocabulary:
                idx = np.where(topics.vocabulary == word)[0]
                topic = np.argmax(topics.topic_terms[:, idx])
                point_colors.append(colors[topic])
            else:
                point_colors.append('white')

        plt.scatter(P[:, 0], P[:, 1], c=point_colors, edgecolors='k')
        plt.title(name + " annotated")
        for label, x, y in zip(labels, P[:, 0], P[:, 1]):
            plt.annotate(label, xy=(x+0.06, y+0.03), xytext=(0, 0), textcoords='offset points')

        plt.figure(figsize=(18, 10))
        plt.scatter(P[:, 0], P[:, 1], c=point_colors, edgecolors='k')
        plt.title(name)
    else:
        # Otherwise, calculate from CBOW
        similar_words = {search_term: [model.id2word[idx] for idx in
                                       model.distance_matrix[model.word2id[search_term]-1].argsort()[1:6]+1]
                         for search_term in search_terms}

        words = sum([[k] + v for k, v in similar_words.items()], [])
        words_ids = [model.word2id[w] for w in words]

        weights = model.model.get_weights()[0]
        weights = weights[1:]

        word_vectors = np.array([weights[idx] for idx in words_ids])

        tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
        np.set_printoptions(suppress=True)
        T = tsne.fit_transform(word_vectors)
        labels = words

        point_colors = []
        colors = ['black', 'red', 'limegreen', 'blue', 'magenta']
        for word in words:
            if word in topics.vocabulary:
                idx = np.where(topics.vocabulary == word)[0]
                topic = np.argmax(topics.topic_terms[:, idx])
                point_colors.append(colors[topic])
            else:
                point_colors.append('white')

        plt.figure(figsize=(18, 10))
        plt.scatter(T[:, 0], T[:, 1], c=point_colors, edgecolors='k')
        plt.title(name + " annotated")
        for label, x, y in zip(labels, T[:, 0], T[:, 1]):
            plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

        plt.figure(figsize=(18, 10))
        plt.scatter(T[:, 0], T[:, 1], c=point_colors, edgecolors='k')
        plt.title(name)


def main():
    # Input and output directory
    in_dir = "Bioanalysis Zone/"
    out_dir = "Data/"

    # Preprocess all HTML files
    process_all_html(in_dir, out_dir + "/emails")

    # Load clean corpus with sklearn.datasets
    data = skdata.load_files(out_dir)
    # Convert to numpy array and remove empty emails
    data_list = np.array([s.decode('ascii') for s in data['data'] if len(s) > 2])

    # Topic modelling for 5 topics
    topics = LDATopicModelling(data, total_topics=5, write_to_file=True)

    # Featurization using CBOW
    cbow = CBOWGenerator(data_list, load_weights=True, training_epochs=50)
    plot_model(cbow, topics, use_gensim=False, name="cbow")

    # Featurization using Word2Vec in Gensim
    w2v = Word2VecGenerator(data_list)
    plot_model(w2v.model, topics, name="w2v")

    # Featurization using FastText
    fast = FastTextGenerator(data_list)
    plot_model(fast.model, topics, name="fast")

    plt.show()

if __name__ == "__main__":
    main()
