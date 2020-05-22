"""
Each Trainer class type extends the corresponding Parser class type
"""
# pylint: disable=bad-continuation, too-many-arguments

from ogm.parser import TextParser


class TextTrainer(TextParser):
    """
    Utility functions for training text-based models
    Extends `TextParser`
    """

    model_types = {"lda"}

    def __init__(self, language="english"):
        super(TextTrainer, self).__init__(language=language)
        self.model_type = None
        self.model = None

    def load_model(self, model_type, input_path):
        """
        Load a model from `input_path` to the `self.model` attribute.
        Will raise a KeyError if the model type is not supported
        """

        if model_type not in self.model_types:
            raise KeyError("Unsupported model type")

        if model_type == "lda":
            from gensim.models import LdaModel

            self.model = LdaModel.load(input_path)
            self.model_type = model_type
        else:
            raise ValueError(
                "Specified model type: " + model_type + " is not yet implemented."
            )

    def train_lda(
        self,
        key=None,
        n_topics=100,
        multicore=True,
        n_workers=4,
        passes=10,
        output_path="./lda.model",
    ):
        """
        Train a gensim LDA model with the specified parameters.
        Will save this model to disk at the specified `output_path`.
        Will generate a gensim dictionary and BoW structure on the data
        with header `key`, unless this has already been done
        """
        if key is None and self.dictionary is None:
            raise ValueError("Please specify a key to generate dictionary from")

        if self.dictionary is None:
            self.make_dict_and_corpus(key)

        if not multicore:
            from gensim.models import LdaModel

            lda_model = LdaModel(
                corpus=self.corpus,
                num_topics=n_topics,
                id2word=self.dictionary,
                passes=passes,
            )

        else:
            from gensim.models import LdaMulticore

            lda_model = LdaMulticore(
                corpus=self.corpus,
                num_topics=n_topics,
                id2word=self.dictionary,
                passes=passes,
                workers=n_workers,
            )

        lda_model.save(output_path)
        self.model = lda_model
        self.model_type = "lda"

    def predict_lda(self, input_data):
        """
        Use an internal LDA model to make a prediction about the topic distribution
        in the `input_data` document, which is just a string
        """
        if self.model is None:
            raise ValueError("No trained model in this TextTrainer")

        if self.model_type != "lda":
            raise ValueError("Model type is " + self.model_type)

        from nltk.stem import WordNetLemmatizer, SnowballStemmer
        from gensim.parsing.preprocessing import STOPWORDS
        from gensim.utils import simple_preprocess

        result = []
        stemmer = SnowballStemmer(self.lang)
        # Run simple_preprocess and generate a list of tokens from this document
        for token in simple_preprocess(input_data):

            # Ignore stopwords and short words, stem/lemmatize the rest
            if token not in STOPWORDS and len(token) > 3:
                lemm_stem = stemmer.stem(WordNetLemmatizer().lemmatize(token, pos="v"))

                # Append this result to list of words
                result.append(lemm_stem)

        # Use list of words to predict topics
        doc_vector = self.model.id2word.doc2bow(result)
        return self.model[doc_vector]

    def train_ldaseq(
        self,
        key=None,
        sort_key=None,
        n_topics=100,
        passes=10,
        seq_counts=None,
        chain_variance=0.005,
        output_path="./ldaseq.model",
    ):
        """
        Train a gensim Sequential LDA model. Multicore is currently not supported in gensim,
        so this model will take quite a bit longer to train. `seq_counts` is a list of integers
        indicating the number of documents in each time slice. Parser will optionally sort its
        data by the key specified in `sort_key`
        """

        if seq_counts is None:
            raise ValueError("Please specify a list of integers for seq_counts")

        if key is None and self.dictionary is None:
            raise ValueError("Please specify a key to generate dictionary from")

        if self.dictionary is None:
            self.make_dict_and_corpus(key)

        if sort_key is not None:
            self.data.sort(key=lambda r: r[sort_key])

        from gensim.models.ldaseqmodel import LdaSeqModel

        lda_model = LdaSeqModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            time_slice=
        )

        lda_model.save(output_path)
        self.model = lda_model
        self.model_type = "ldaseq"
