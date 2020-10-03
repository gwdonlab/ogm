"""
Parser class
"""
# pylint: disable=bad-continuation, too-many-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, too-many-nested-blocks
import pickle
import csv
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
        self.earliest_data = None
        self.has_datetime = None
        self._index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index >= len(self.data):
            self._index = -1
            raise StopIteration
        else:
            return self.data[self._index]

    def __reversed__(self):
        return self.data[::-1]

    def parse_file(self, filepath, sheet=0, encoding="utf8", pdf_append=True):
        """
        Parse supported file types.
        If parsing an Excel file, optionally specify a `sheet` to be read from the workbook.
        If parsing a delimited file (.tsv, .csv), optionally specify an `encoding`.
        R must be installed to parse .rds files.
        If `pdf_append` is False, `self.data` will be overwritten by incoming PDF.
        """

        if filepath.endswith(".tsv"):
            data_dicts = []
            data_temp = []

            with open(filepath, "r", encoding=encoding) as csvin:
                csvin = csv.reader(csvin, delimiter="\t")

                for row in csvin:
                    data_temp.append(row)

            for row in data_temp[1:]:

                # Skip row if it's empty
                if len(row) == 0:
                    continue

                data_row = {}
                for heading in data_temp[0]:
                    data_row[heading] = row[data_temp[0].index(heading)]

                data_dicts.append(data_row)

            self.data = data_dicts

        elif filepath.endswith(".csv"):
            data_dicts = []
            data_temp = []

            with open(filepath, "r", encoding=encoding) as csvin:
                csvin = csv.reader(csvin, delimiter=",")

                for row in csvin:
                    data_temp.append(row)

            for row in data_temp[1:]:

                # Skip row if it's empty
                if len(row) == 0:
                    continue

                data_row = {}
                for heading in data_temp[0]:
                    data_row[heading] = row[data_temp[0].index(heading)]

                data_dicts.append(data_row)

            self.data = data_dicts

        elif filepath.endswith(".xlsx"):
            import openpyxl

            data_dicts = []
            header_list = []

            open_wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True,)
            sheet_name = open_wb.sheetnames[sheet]
            open_sheet = open_wb[sheet_name]

            for row_id, row in enumerate(open_sheet.rows):
                new_entry = {}
                for col_id, cell in enumerate(row):
                    value = cell.value

                    try:
                        if row_id == 0:
                            if value is None:
                                break
                            header_list.append(value)
                        else:
                            if value is None:
                                new_entry[header_list[col_id]] = ""
                            else:
                                new_entry[header_list[col_id]] = value
                    except IndexError:
                        break

                if row_id > 0:
                    data_dicts.append(new_entry)

            self.data = data_dicts

        elif filepath.endswith(".rds"):
            import rpy2.robjects as robjects
            from rpy2.robjects import pandas2ri

            pandas2ri.activate()
            read_rds = robjects.r["readRDS"]
            dataframe_temp = read_rds(filepath)
            self.data = dataframe_temp.to_dict("records")

        elif filepath.endswith(".pkl"):
            self.import_self(filepath)

        elif filepath.endswith(".pdf"):
            from tika import parser

            raw = parser.from_file(filepath)
            if pdf_append:
                self.data.append({"text": raw["content"]})
            else:
                self.data = [{"text": raw["content"]}]

        else:
            raise IOError("Unsupported file type")

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
        Alias for `get_attribute_list`. Use this function in the future
        """
        print("WARNING: get_texts is deprecated. Use get_attribute_list instead")
        return self.get_attribute_list(key)

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
        if self.has_datetime is not None:
            print(
                "WARNING: TextParser already has datetime attribute at",
                self.has_datetime,
            )

        if not self.data:
            raise ValueError("Parse a text file first")

        if key not in self.data[0].keys():
            raise KeyError("Time key isn't in this dataset")

        if key_to_add in self.data[0].keys() and not overwrite:
            raise KeyError(
                "Trying to add a pre-existing key. Set 'overwrite' to True to ignore"
            )

        if self.earliest_data is None:
            self.earliest_data = datetime.datetime.now()

        for item in self.data:
            item[key_to_add] = datetime.datetime.strptime(item[key], data_format)
            if item[key_to_add] < self.earliest_data:
                self.earliest_data = item[key_to_add]

        self.has_datetime = key_to_add

    def find_earliest_data(self, key, data_format):
        """
        Scans data with timestamp `key` formatted with `data_format`
        to find the earliest-occurring datd point
        """
        self.earliest_data = datetime.datetime.now()

        for item in self.data:
            timestamp = datetime.datetime.strptime(item[key], data_format)

            if timestamp < self.earliest_data:
                self.earliest_data = timestamp

    def __str__(self):
        stub = "Parser Object\n\tDocuments: %d" % len(self.data)
        if self.data:
            stub += "\n\tHeaders: " + str(self.data[0].keys())
            stub += "\n\tLanguage: " + self.lang
            stub += "\n\tStemmed: " + str(self.stemmed)
        return stub

    def merge_data(self, list_of_textparsers):
        """
        For each `TextParser` in `list_of_textparsers`, merge its `data` attribute into this
        `TextParser`'s `data` attribute. Will check to ensure data doesn't have different headers
        and hasn't been stemmed. These operations are considered unsafe and will throw exceptions.
        """

        for parser in list_of_textparsers:

            # Warn user if a passed TextParser is empty
            if not parser.data:
                print("WARNING: Found empty TextParser. Skipping...")
                continue

            # Make sure you aren't duplicating a parser
            if parser is self:
                raise ValueError("Tried to merge self with self")

            # Make sure parsers aren't stemmed
            if parser.stemmed or self.stemmed:
                raise ValueError(
                    "Merging TextParsers that have already been stemmed is unsafe"
                )

            # Make sure parsers have the same headers
            for header in parser.data[0].keys():
                if header not in self.data[0]:
                    raise KeyError(
                        "Merging TextParsers with different headers is unsafe: "
                        + "Found a key in one of the arguments that isn't in this parser"
                    )

            for header in self.data[0].keys():
                if header not in parser.data[0]:
                    raise KeyError(
                        "Merging TextParsers with different headers is unsafe: "
                        + "Found a key in this parser that isn't in one of the arguments"
                    )

            # Append each data point from parser into this TextParser
            for datum in parser.data:
                self.data.append(datum)

    def plot_data_quantities(
        self,
        key,
        data_format,
        days_interval,
        start_date=None,
        end_date=None,
        normalize=False,
        show_plot=True,
        color=None,
        plot_title="Quantity of data in time frames",
    ):
        """
        Makes a matplot graph of of the numbers of posts over time. Requires a `key` where
        the timestamps are stored, a `data_format` to allow `datetime` to parse the timestamp,
        and a `days_interval` to tell how large each time interval is.
        Starts at earliest found timestamp, unless `start_date` is specified with `data_format`.
        Ends at current date, unless `end_date` is specified with `data_format`.
        Will run `add_datetime_attribute` with key "__added_datetime"
        if this isn't manually run earlier. You can choose to automatically display the
        generated plot or not with the `show_plot` flag. Returns the x and y axes.
        The `color` argument must be `None` or a valid matplotlib color code
        """

        import matplotlib.pyplot as plt

        if not self.has_datetime:
            self.add_datetime_attribute(key, data_format, "__added_datetime")

        if start_date is None:
            beginning = self.earliest_data
        else:
            beginning = datetime.datetime.strptime(start_date, data_format)

        if end_date is None:
            end = datetime.datetime.now()
        else:
            end = datetime.datetime.strptime(end_date, data_format)

        # Sort data chronologically
        self.data.sort(key=lambda x: x[self.has_datetime])

        # Initialize empty logistics structures
        x_axis_labels = []
        y_axis_quantities = []
        left_off_at = 0

        # If the user didn't give a start date, we're guaranteed that we should start at index 0
        if start_date is not None:
            try:
                # Find correct index to start counting posts at
                while self.data[left_off_at][self.has_datetime] < beginning:
                    left_off_at += 1

            except IndexError:
                raise ValueError("Couldn't find any posts in specified time frame")

        # Iterate through the data and tally up how many posts are in each bucket
        # This runs in O(n) time:
        # left_off_at jumps from timeslice to timeslice,
        # while the secondary start_index iterator loops through posts within current timeslice
        while beginning < end:
            end_of_timeslice = min(
                beginning + datetime.timedelta(days=days_interval), end
            )
            x_axis_labels.append(beginning.strftime("%Y-%m-%d"))
            start_index = left_off_at
            quant_in_timeslice = 0
            while (
                start_index < len(self.data)
                and self.data[start_index][self.has_datetime] < end_of_timeslice
            ):
                quant_in_timeslice += 1
                start_index += 1
            left_off_at = start_index
            y_axis_quantities.append(quant_in_timeslice)
            beginning = end_of_timeslice

        p_title = plot_title
        if normalize:
            y_ax = [x / sum(y_axis_quantities) for x in y_axis_quantities]
            plt.bar(x_axis_labels, y_ax, color=color)
            p_title += " (data normalized)"
            plt.ylabel("Fraction of total documents")
        else:
            plt.bar(x_axis_labels, y_axis_quantities, color=color)
            plt.ylabel("Number of documents")

        plt.title(p_title)
        plt.xlabel("Start day of time frame")

        if show_plot:
            plt.show()

        return x_axis_labels, y_axis_quantities

    def get_attribute_list(self, key):
        """
        Return a list of the data contained under the header `key`
        """
        return [x[key] for x in self.data]

    def write_data(
        self, filepath, delimiter=",", write_headers=True, encoding="iso8859"
    ):
        """
        Deprecated function
        """
        print(
            "WARNING: write_data will be removed in a future version. Use write_csv instead"
        )

        self.write_csv(
            filepath,
            delimiter=delimiter,
            write_headers=write_headers,
            encoding=encoding,
        )

    def write_csv(
        self, filepath, delimiter=",", write_headers=True, encoding="iso8859"
    ):
        """
        Write data out to a csv-like file at `filepath`.
        The delimiter (comma by default) can be anything
        """

        headers = self.data[0].keys()

        with open(filepath, "w", newline="", encoding=encoding) as outfile:
            filewriter = csv.DictWriter(
                outfile, delimiter=delimiter, fieldnames=headers
            )

            if write_headers:
                filewriter.writeheader()

            for item in self.data:
                filewriter.writerow(item)

    def merge_words(self, key, new_key=None):
        """
        For all data points, merges a list of words in `key` into a string at `key` instead,
        unless `new_key` is not `None`
        """
        if new_key is None:
            dest = key
        else:
            dest = new_key

        for item in self.data:
            new_string = " ".join(item[key])
            item[dest] = new_string

    def filter_doc_length(self, key, min_length, complement=False):
        """
        Removes all documents at `key` with length shorter than `min_length`
        """
        if not complement:
            self.data = [x for x in self.data if len(x[key]) >= min_length]
        else:
            self.data = [x for x in self.data if len(x[key]) < min_length]


class ImageParser:
    """
    *WIP:* Read image data from a variety of sources and perform various processing tasks on it
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
