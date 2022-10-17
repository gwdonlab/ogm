"""
Parser class
"""
# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, too-many-nested-blocks
import datetime
import pandas as pd


class TextParser:
    """
    Read text data from a variety of sources and perform various processing tasks on it
    """

    def __init__(self, language="english"):
        """
        Supported languages depend on which methods are being used.
        Use NLTK's SnowballStemmer for list of supported languages
        """
        self.lang = language
        self.stemmed = False
        self.data = None
        self.dictionary = None
        self.corpus = None
        self._index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index >= self.data.shape[0]:
            self._index = -1
            raise StopIteration

        return self.data[self._index]

    def __getitem__(self, ind):
        """
        If subscripting with an int, it will be treated as a list index in `self.data`.
        """
        if isinstance(ind, int):
            return self.data.iloc[ind]

        raise KeyError("Unknown TextParser subscript received:", ind)

    def parse_file(self, filepath, sheet=0, encoding="utf8", id_col=None):
        """
        Parse supported file types.
        If parsing an Excel file, optionally specify a `sheet` to be read from the workbook.
        If parsing a delimited file (.tsv, .csv), optionally specify an `encoding`.
        Optionally, specify a column to be set as DataFrame index.
        Calling this function twice will overwrite previous data.
        """

        if filepath.endswith(".tsv"):
            self.data = pd.read_csv(filepath, sep="\t", encoding=encoding)

        elif filepath.endswith(".csv"):
            self.data = pd.read_csv(filepath, encoding=encoding)

        elif filepath.endswith(".xlsx"):
            self.data = pd.read_excel(filepath, sheet_name=sheet)

        elif filepath.endswith(".pkl"):
            self.data = pd.read_pickle(filepath)

        elif filepath.endswith(".ft"):
            self.data = pd.read_feather(filepath)

        else:
            raise IOError("Unsupported file type")

        if id_col is not None:
            self.data.set_index(id_col, verify_integrity=True, inplace=True)

    def remove_words(self, col, remove_words):
        """
        Remove the words in `remove_words` in `data` at the column `col`.
        `remove_words` should be specified as a `set` object.
        If the data hasn't been stemmed, this does string-matching.
        If the data has been stemmed, this will search through the list of word stems
        """

        if self.data is None:
            raise ValueError("Parse a text file first!")

        if self.stemmed:
            self.data[col] = self.data[col].apply(
                lambda cell: [word for word in cell if word not in remove_words]
            )
        else:
            for word in remove_words:
                self.data[col] = self.data[col].str.replace(word, "", regex=False)

    def replace_words(self, col, replacement_map):
        """
        Replace words in `data` at `col` according to mappings in the `replacement_map` dict.
        For example, set this dict to `{"cat" : "dog"}` to replace all instances
        of "cat" with "dog".
        """

        if self.data is None:
            raise ValueError("Parse a text file first!")

        if self.stemmed:
            self.data[col] = self.data[col].apply(
                lambda cell: [
                    replacement_map[word] if word in replacement_map else word for word in cell
                ]
            )
        else:
            for word in replacement_map:
                self.data[col] = self.data[col].str.replace(replacement_map[word], "", regex=False)

    def lemmatize_stem_words(self, col, pos="v", min_len=3):
        """
        Stem and lemmatize words in `data` at the column `col` using
        NLTK's SnowballStemmer and WordNetLemmatizer.
        Also converts words to lowercase and expands contractions.
        Stemming means removing suffixes and lemmatizing means converting
        all words to first-person, present tense when possible.

        This behavior can be modified by passing an NLTK-accepted POS code.

        Ignores unknown words or words unable to be altered.
        """
        from gensim.parsing.preprocessing import STOPWORDS
        from gensim.utils import simple_preprocess
        from ogm.utils import lemmatize_string, stem_string, fix_contractions

        if self.data is None:
            raise ValueError("Parse a text file first!")

        if self.stemmed:
            raise ValueError("Data has already been lemmatized and stemmed")

        def lemm_stem_str(input_str):
            result = []

            # Run simple_preprocess and generate a list of tokens from this document
            for token in simple_preprocess(input_str, min_len=min_len):

                # Ignore stopwords and short words, stem/lemmatize the rest
                if token not in STOPWORDS:
                    lemm_stem = stem_string(
                        lemmatize_string(token, do_not_tokenize=True, pos=pos),
                        not_tokenized=False,
                        language=self.lang,
                    )
                    result.append(lemm_stem[0])

            return result

        # Handle English contractions
        if self.lang == "english":
            no_contractions = self.data[col].apply(fix_contractions)
        else:
            no_contractions = self.data[col]

        self.data[col] = no_contractions.apply(lemm_stem_str)
        self.stemmed = True

    def make_dict_and_corpus(self, col, filter_vocab_above_thresh=None):
        """
        Will populate the `dictionary` and `corpus` attributes without training a model
        """
        from gensim.corpora import Dictionary

        temp = self.data[col].tolist()
        self.dictionary = Dictionary(temp)
        if filter_vocab_above_thresh:
            self.dictionary.filter_extremes(no_above=filter_vocab_above_thresh)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in temp]

    def filter_data(self, col, acceptable_vals, complement=False):
        """
        Filter dataset so that only items in `acceptable_vals` appear in the data's `col` column.
        `acceptable_vals` should be specified as a `set` object.
        This operation can be complemented by setting `complement` to True.
        This can be called multiple times, but data will be deleted if it doesn't match the filter.
        Will return the number of entries removed
        """
        if self.data is None:
            raise ValueError("Parse a text file first!")

        if not complement:
            temp = self.data[self.data[col].isin(acceptable_vals)]
        else:
            temp = self.data[~self.data[col].isin(acceptable_vals)]

        items_removed = self.data.shape[0] - temp.shape[0]
        self.data = temp

        return items_removed

    def filter_within_time_range(
        self, col, input_format, start, end, data_format=None, complement=False
    ):
        """
        Filter dataset so that items with a time attribute are restricted by the time frame from
        `start` to `end`. Will include `start` but exclude `end`.
        `data_format` specifies how the date appears in the data. If `None`, this will be inferred.
        `input_format` specifies how the date appears in the `start` and `end` arguments.
        Operation can be complemented by setting `complement` to True.
        Will return the number of entries removed
        """
        if self.data is None:
            raise ValueError("Parse a text file first!")

        date_f = datetime.datetime.strptime(end, input_format)
        date_i = datetime.datetime.strptime(start, input_format)
        temp = []

        def determine_date(input_dt):
            return input_dt >= date_i and input_dt < date_f

        if data_format is not None:
            if not complement:
                temp = self.data[
                    self.data[col]
                    .apply(lambda x: datetime.datetime.strptime(x, data_format))
                    .apply(determine_date)
                ]
            else:
                temp = self.data[
                    ~self.data[col]
                    .apply(lambda x: datetime.datetime.strptime(x, data_format))
                    .apply(determine_date)
                ]
        else:
            import dateutil.parser as dp

            if not complement:
                temp = self.data[self.data[col].apply(dp.parse).apply(determine_date)]
            else:
                temp = self.data[~self.data[col].apply(dp.parse).apply(determine_date)]

        items_removed = self.data.shape[0] - temp.shape[0]
        self.data = temp
        return items_removed

    def __str__(self):
        stub = "Parser Object\n\tDocuments: %d" % self.data.shape[0]
        if self.data is not None:
            stub += "\n\tColumns: " + str(list(self.data.columns))
            stub += "\n\tLanguage: " + self.lang
            stub += "\n\tStemmed: " + str(self.stemmed)
        return stub

    def get_attribute_list(self, col):
        """
        Return a list of the data contained under the header `col`
        """
        return self.data[col].tolist()

    def merge_words(self, col, new_col=None):
        """
        For all data points, merges a list of words in `col` into a string at `col` instead,
        unless `new_col` is not `None`
        """
        if new_col is None:
            dest = col
        else:
            dest = new_col

        self.data[dest] = self.data[col].apply(lambda str_list: " ".join(str_list))

    def filter_doc_length(self, col, min_length, complement=False):
        """
        Removes all documents at `col` with length shorter than `min_length`
        """
        if complement:
            self.data = self.data[self.data[col].str.len() < min_length]
        else:
            self.data = self.data[self.data[col].str.len() >= min_length]
