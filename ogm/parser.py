"""
Parser class
"""
# pylint: disable=bad-continuation, too-many-arguments
import pickle
import csv
import gc
import xlrd
import datetime


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
        self.data = []
        self.dictionary = None
        self.corpus = None

    def parse_excel(self, filepath, sheet):
        """
        Parse the Sheet `sheet` (0-indexed) in the Excel file at
        `filepath` into an internal dict list
        """
        data_dicts = []

        open_sheet = xlrd.open_workbook(filepath).sheet_by_index(sheet)
        for row in range(1, open_sheet.nrows):
            data_row = {}
            for heading in open_sheet.row_values(0):
                data_row[heading] = open_sheet.row_values(row)[
                    open_sheet.row_values(0).index(heading)
                ]

            data_dicts.append(data_row)

        self.data = data_dicts

    def parse_tsv(self, filepath, encoding="utf8"):
        """
        Parse the tsv file at `filepath` into an internal dict list.
        Optionally, specify the document's encoding.
        Will assume UTF-8 by default
        """
        data_dicts = []
        data_temp = []

        with open(filepath, "r", encoding=encoding) as tsvin:
            tsvin = csv.reader(tsvin, delimiter="\t")

            for row in tsvin:
                data_temp.append(row)

        for row in data_temp[1:]:
            data_row = {}
            for heading in data_temp[0]:
                data_row[heading] = row[data_temp[0].index(heading)]

            data_dicts.append(data_row)

        self.data = data_dicts

    def parse_csv(self, filepath, encoding="utf8"):
        """
        Parse the csv file at `filepath` into an internal dict list.
        Optionally, specify the document's encoding.
        Will assume UTF-8 by default
        """
        data_dicts = []
        data_temp = []

        with open(filepath, "r", encoding=encoding) as csvin:
            csvin = csv.reader(csvin)

            for row in csvin:
                data_temp.append(row)

        for row in data_temp[1:]:
            data_row = {}
            for heading in data_temp[0]:
                data_row[heading] = row[data_temp[0].index(heading)]

            data_dicts.append(data_row)

        self.data = data_dicts

    def parse_pdf(self, filepath):
        """
        Read the text in the PDF at `filepath` into `self.data` -- not recommended
        Uses the tika package
        """
        from tika import parser

        raw = parser.from_file(filepath)
        self.data = [{"text": raw["content"]}]

    def parse_add_pdf(self, filepath):
        """
        Just like `parse_pdf`, but will add the text to the data rather than replacing it
        """
        from tika import parser

        raw = parser.from_file(filepath)
        self.data.append({"text": raw["content"]})

    def export_self(self, outpath="./output.pkl"):
        """
        Dumps all instance variables into a pickle file
        """
        with open(outpath, "wb") as pickle_out:
            pickle.dump(self.__dict__, pickle_out)

    def import_self(self, inpath="./output.pkl"):
        """
        Loads instance attributes from a pickle file; will prioritize
        attributes found in the pickle file over preset ones
        """
        with open(inpath, "rb") as pickle_in:
            temp = pickle.load(pickle_in)

        self.__dict__.update(temp)

    def remove_words(self, key, remove_words):
        """
        Remove the words in `remove_words` in `data` at the dict key `key`.
        `remove_words` should be specified as a `set` object.
        If the data hasn't been stemmed, this does string-matching.
        If the data has been stemmed, this will search through the list of word stems
        """

        if not self.data:
            raise ValueError("Please parse a text file first!")

        if key not in self.data[0].keys():
            raise KeyError("Trying to parse text on a key that doesn't exist")

        occurrences = 0
        new_dicts = []

        # Loop through data dictionaries
        for data_dict in self.data:
            new_dict = {}

            # Loop through each entry in the data dictionary
            for dict_key in data_dict:

                # If this is true, then this is the data to be cleaned/data is a word list
                if dict_key == key and self.stemmed:
                    new_dict[dict_key] = list(
                        filter(lambda word: word not in remove_words, data_dict[key])
                    )
                    occurrences += len(data_dict[dict_key]) - len(new_dict[dict_key])

                # If this is true, then this is the data to be cleaned/data is a string
                elif dict_key == key:
                    curr_string = data_dict[key]
                    for word in remove_words:
                        curr_string = curr_string.replace(word, "")
                        occurrences += int(
                            (len(data_dict[dict_key]) - len(curr_string)) / len(word)
                        )
                    new_dict[dict_key] = curr_string

                # Otherwise, this data should just be copied over
                else:
                    new_dict[dict_key] = data_dict[dict_key]

            new_dicts.append(new_dict)

        self.data = new_dicts
        gc.collect()
        return occurrences

    def replace_words(self, key, replacement_map):
        """
        Replace words in `data` at `key` according to mappings in the `replacement_map` dict.
        For example, set this dict to `{"cat" : "dog"}` to replace all instances
        of "cat" with "dog".
        """

        if not self.data:
            raise ValueError("Please parse a text file first!")

        if key not in self.data[0].keys():
            raise KeyError("Trying to parse text on a key that doesn't exist")

        new_dicts = []

        # Loop through data dictionaries
        for data_dict in self.data:
            new_dict = {}

            # Loop through each entry in the data dictionary
            for dict_key in data_dict:

                # If this is true, then this is the data to be cleaned/data is a word list
                if dict_key == key and self.stemmed:
                    new_word_list = []
                    for word in data_dict[dict_key]:
                        if word in replacement_map:
                            new_word_list.append(replacement_map[word])
                        else:
                            new_word_list.append(word)

                    new_dict[dict_key] = new_word_list

                # If this is true, then this is the data to be cleaned/data is a string
                elif dict_key == key:
                    curr_string = data_dict[dict_key]
                    for word in replacement_map:
                        curr_string = curr_string.replace(word, replacement_map[word])
                    new_dict[dict_key] = curr_string

                # Otherwise, this data should just be copied over
                else:
                    new_dict[dict_key] = data_dict[dict_key]

            new_dicts.append(new_dict)

        self.data = new_dicts
        gc.collect()

    def lemmatize_stem_words(self, key):
        """
        Stem and lemmatize words in `data` at the dict key `key` using
        NLTK's SnowballStemmer and WordNetLemmatizer.
        Also converts words to lowercase.
        Stemming means removing suffixes and lemmatizing means converting
        all words to first-person, present tense when possible.
        Ignores unknown words or words unable to be altered.
        """
        try:
            from nltk.stem import WordNetLemmatizer, SnowballStemmer
            from gensim.parsing.preprocessing import STOPWORDS
            from gensim.utils import simple_preprocess
        except:
            raise ImportError("This method requires the gensim and nltk packages.")

        if not self.data:
            raise ValueError("Please parse a text file first!")

        if key not in self.data[0].keys():
            raise KeyError("Trying to parse text on a key that doesn't exist")

        if self.stemmed:
            raise ValueError("Data has already been lemmatized and stemmed")

        data_dicts = []

        # Create Stemmer object
        stemmer = SnowballStemmer(self.lang)

        # Iterate through self.data
        for data_dict in self.data:
            new_dict = {}
            for dict_key in data_dict:

                # This key is the one to be operated on
                if dict_key == key:
                    result = []

                    # Run simple_preprocess and generate a list of tokens from this document
                    for token in simple_preprocess(data_dict[key], min_len=3):

                        # Ignore stopwords and short words, stem/lemmatize the rest
                        if token not in STOPWORDS:
                            lemm_stem = stemmer.stem(
                                WordNetLemmatizer().lemmatize(token, pos="v")
                            )

                            # Append this result to list of words
                            result.append(lemm_stem)

                    # Put this list of words back into the data dict
                    new_dict[dict_key] = result

                # This key is metadata and should just be copied
                else:
                    new_dict[dict_key] = data_dict[dict_key]

            data_dicts.append(new_dict)

        self.data = data_dicts
        self.stemmed = True
        gc.collect()

    def make_dict_and_corpus(self, key):
        """
        Will populate the `dictionary` and `corpus` attributes without training a model
        """
        from gensim.corpora import Dictionary

        temp = [entry[key] for entry in self.data]
        self.dictionary = Dictionary(temp)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in temp]

    def get_texts(self, key):
        """
        Return a list of texts with header `key`
        """
        return [x[key] for x in self.data]

    def filter_data(self, key, acceptable_vals, complement=False):
        """
        Filter dataset so that only items in `acceptable_vals` appear in the data's `key` heading.
        `acceptable_vals` should be specified as a `set` object.
        This operation can be complemented by setting `complement` to True.
        This can be called multiple times, but data will be deleted if it doesn't match the filter.
        Will return the number of entries removed
        """
        if not self.data:
            raise ValueError("Parse a text file first")

        if key not in self.data[0].keys():
            raise KeyError("Trying to filter by a key that isn't in the dataset")

        temp = []
        if not complement:
            temp = [x for x in self.data if x[key] in acceptable_vals]
        else:
            temp = [x for x in self.data if x[key] not in acceptable_vals]

        items_removed = len(self.data) - len(temp)
        self.data = temp

        return items_removed

    def filter_within_time_range(
        self, key, data_format, input_format, start, end, complement=False
    ):
        """
        Filter dataset so that items with a time attribute are restricted by the time frame from
        `start` to `end`. Will include `start` but exclude `end`.
        `data_format` specifies how the date appears in the data.
        `input_format` specifies how the date appears in the `start` and `end` arguments.
        Operation can be complemented by setting `complement` to True.
        Will return the number of entries removed
        """
        if not self.data:
            raise ValueError("Parse a text file first")

        if key not in self.data[0].keys():
            raise KeyError("Time key isn't in this dataset")

        date_f = datetime.datetime.strptime(end, input_format)
        date_i = datetime.datetime.strptime(start, input_format)
        temp = []

        if not complement:
            temp = [
                x
                for x in self.data
                if (
                    datetime.datetime.strptime(x[key], data_format) >= date_i
                    and datetime.datetime.strptime(x[key], data_format) < date_f
                )
            ]
        else:
            temp = [
                x
                for x in self.data
                if (
                    datetime.datetime.strptime(x[key], data_format) >= date_i
                    and datetime.datetime.strptime(x[key], data_format) < date_f
                )
            ]

        items_removed = len(self.data) - len(temp)
        self.data = temp
        return items_removed

    def add_datetime_attribute(self, key, data_format, key_to_add, overwrite=False):
        """
        Adds a key to the data list called `key_to_add`. This key will hold a `datetime`
        object which is built from the string at `key` written in the format `data_format`.
        See documentation for `datetime` to learn how to specify this format.
        Useful for chronological comparisons in Python
        """
        if not self.data:
            raise ValueError("Parse a text file first")

        if key not in self.data[0].keys():
            raise KeyError("Time key isn't in this dataset")

        if key_to_add in self.data[0].keys() and not overwrite:
            raise KeyError(
                "Trying to add a key with data in it. Set 'overwrite' to True to peform this operation"
            )

        for item in self.data:
            item[key_to_add] = datetime.datetime.strptime(item[key], data_format)

    def __str__(self):
        stub = "Parser Object\n\tDocuments: %d" % len(self.data)
        if self.data:
            stub += "\n\tHeaders: " + str(self.data[0].keys())
            stub += "\n\tLanguage: " + self.lang
            stub += "\n\tStemmed: " + str(self.stemmed)
        return stub


class ImageParser:
    """
    WIP: Read image data from a variety of sources and perform various processing tasks on it
    """

    def __init__(self):
        self.data = []

    def parse_csv(self, filepath, labeled=True):
        """
        Parse the csv file at `filepath` into an internal dict list.
        Assumes each row is a label followed by pixel values.
        """
        data_dicts = []

        with open(filepath, "r") as csvin:
            csvin = csv.reader(csvin)

            for row in csvin:
                try:
                    numeric_data = [float(x) for x in row]
                except ValueError:
                    continue
                data_point = {}
                if labeled:
                    data_point["label"] = int(numeric_data[0])
                    data_point["pixels"] = numeric_data[1:]
                else:
                    data_point["pixels"] = numeric_data
                data_dicts.append(data_point)

        self.data = data_dicts

    def export_self(self, outpath="./output.pkl"):
        """
        Dumps all instance variables into a pickle file
        """
        with open(outpath, "wb") as pickle_out:
            pickle.dump(self.__dict__, pickle_out)

    def import_self(self, inpath="./output.pkl"):
        """
        Loads instance attributes from a pickle file.
        Will prioritize attributes found in the pickle file over preset ones
        """
        with open(inpath, "rb") as pickle_in:
            temp = pickle.load(pickle_in)

        self.__dict__.update(temp)
