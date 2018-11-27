import string
import collections
import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class KMeansClustering(object):
    def __init__(self, text_data, num_clusters):
        self.dataset = text_data
        self.num_clusters = num_clusters

    def clustering(self):
        """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
        vectorizer = TfidfVectorizer(tokenizer=self.process_text,
                                     stop_words=stopwords.words('english'),
                                     max_df=0.5,
                                     min_df=1,
                                     lowercase=True)

        tf_idf_model = vectorizer.fit_transform(self.dataset)
        km_model = KMeans(n_clusters=self.num_clusters)
        km_model.fit(tf_idf_model)

        clustering = collections.defaultdict(list)

        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)

        return clustering

    def process_text(self, text, stem=True):
        """ Tokenize text and stem words removing punctuation """
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        if stem:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        return tokens


def main():
    categories = ['alt.atheism', 'comp.graphics', 'sci.crypt']
    dataset = fetch_20newsgroups(subset='all',
                                 categories=categories,
                                 shuffle=True)
    k_means_clustering = KMeansClustering(dataset.data, 3)
    clusters = k_means_clustering.clustering()
    print(dict(clusters))


if __name__ == "__main__":
    main()
