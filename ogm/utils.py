"""
Commonly-used utility functions
This is mainly here to reduce code duplication
"""

# pylint: disable=too-many-branches


def text_data_preprocess(setup_dict, output=False, return_parser=False):
    """
    Run preprocessing on text data; `setup_dict` should be in JSON format described in README.
    Set `output` to write processed data into a file and return None.
    Set `return_parser` to return a TextParser rather than a dataframe
    """
    from ogm.parser import TextParser

    # Absolute path to data file
    data_file = setup_dict["data_path"]

    # Optional language attribute in setup
    if "lang" in setup_dict:
        parser = TextParser(language=setup_dict["lang"])
    else:
        parser = TextParser()

    # Optional text encoding attribute in setup
    if "encoding" in setup_dict:
        file_encoding = setup_dict["encoding"]
    else:
        file_encoding = "utf8"

    parser.parse_file(data_file, encoding=file_encoding)
    parser.data.dropna(subset=[setup_dict["text_key"]], inplace=True)
    parser.data.reset_index(drop=True, inplace=True)

    # Optional time frame attribute
    if "time_filter" in setup_dict and "data_format" in setup_dict["time_filter"]:
        parser.filter_within_time_range(
            col=setup_dict["time_filter"]["time_key"],
            data_format=setup_dict["time_filter"]["data_format"],
            input_format=setup_dict["time_filter"]["arg_format"],
            start=setup_dict["time_filter"]["start"],
            end=setup_dict["time_filter"]["end"],
        )
    elif "time_filter" in setup_dict:
        parser.filter_within_time_range(
            col=setup_dict["time_filter"]["time_key"],
            input_format=setup_dict["time_filter"]["arg_format"],
            start=setup_dict["time_filter"]["start"],
            end=setup_dict["time_filter"]["end"],
        )

    # Other attribute filters
    if "attribute_filters" in setup_dict:
        for attr_filter in setup_dict["attribute_filters"]:
            parser.filter_data(attr_filter["filter_key"], set(attr_filter["filter_vals"]))

    if "remove_emoji" in setup_dict:
        for item in parser.data:
            item[setup_dict["text_key"]] = remove_emoji(item[setup_dict["text_key"]])

    if "replace_before_stemming" in setup_dict:
        parser.replace_words(setup_dict["text_key"], setup_dict["replace_before_stemming"])

    if "remove_before_stemming" in setup_dict:
        parser.remove_words(setup_dict["text_key"], set(setup_dict["remove_before_stemming"]))

    parser.lemmatize_stem_words(setup_dict["text_key"])

    if "replace_after_stemming" in setup_dict:
        parser.replace_words(setup_dict["text_key"], setup_dict["replace_after_stemming"])

    if "remove_after_stemming" in setup_dict:
        parser.remove_words(setup_dict["text_key"], set(setup_dict["remove_after_stemming"]))

    if "min_length" in setup_dict:
        parser.filter_doc_length(setup_dict["text_key"], setup_dict["min_length"])

    # Required output path attribute if this arg is true
    if output:
        parser.data.to_csv(setup_dict["output_path"], sep="\t", encoding=file_encoding)

        return None

    if return_parser:
        return parser

    return parser.data


def from_excel_ordinal(ordinal):
    """
    Convert Excel ordinal numbers to a datetime object
    Taken from this StackOverflow answer:
    https://stackoverflow.com/questions/29387137/how-to-convert-a-given-ordinal-number-from-excel-to-a-date
    """
    from datetime import datetime, timedelta

    _epoch0 = datetime(1899, 12, 31)

    if ordinal >= 60:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!

    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)


def remove_emoji(input_text):
    """
    Taken from this Gist:
    https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b#gistcomment-3315605
    """
    import re

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub(r" ", input_text)


def remove_URL(input_text):
    import re

    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r" ", input_text)


def sechidis_stratify(data, classes, ratios, one_hot=False):
    import numpy as np

    """
    `data` is a list of lists: a list of labels, for each sample.
    Each sample's labels should be ints, if they are one-hot encoded, use one_hot=True

    classes is the list of classes each label can take

    ratios is a list, summing to 1, of how the dataset should be split

    Function taken from https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/

    Implements Algorithm 1 from http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
    """

    # one-hot decoding
    if one_hot:
        temp = [[] for _ in range(len(data))]
        indexes, values = np.where(np.array(data).astype(int) == 1)
        for k, v in zip(indexes, values):
            temp[k].append(v)
        data = temp

    # Organize data per label: for each label l, per_label_data[l] contains the list of samples
    # in data which have this label
    per_label_data = {c: set() for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].add(i)

    # number of samples
    size = len(data)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios]
    per_label_subset_sizes = {c: [r * len(per_label_data[c]) for r in ratios] for c in classes}

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))]

    # For each sample in the data set
    while size > 0:
        # Compute |Di|
        lengths = {l: len(label_data) for l, label_data in per_label_data.items()}
        try:
            # Find label of smallest |Di|
            label = min({k: v for k, v in lengths.items() if v > 0}, key=lengths.get)
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error.
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break

        # For each sample with label `label`
        while per_label_data[label]:
            # Select such a sample
            current_id = per_label_data[label].pop()

            subset_sizes_for_label = per_label_subset_sizes[label]
            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(
                subset_sizes_for_label == np.amax(subset_sizes_for_label)
            ).flatten()

            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need
            # of any label
            else:
                largest_subsets = np.argwhere(subset_sizes == np.amax(subset_sizes)).flatten()
                if len(largest_subsets) == 1:
                    subset = largest_subsets[0]
                else:
                    # If there is more than one such subset, choose at random
                    subset = np.random.choice(largest_subsets)

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is one fewer sample to distribute
            size -= 1
            # The selected subset needs one fewer sample
            subset_sizes[subset] -= 1

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1

            # Remove the sample from the dataset, meaning from all per_label dataset created
            for l, label_data in per_label_data.items():
                if current_id in label_data:
                    label_data.remove(current_id)

    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [[data[i] for i in strat] for strat in stratified_data_ids]

    # Return both the stratified indexes, to be used to sample
    # `features` associated with your labels
    # Also the stratified labels dataset
    return stratified_data_ids, stratified_data


def lemmatize_string(words, not_tokenized=True, do_not_tokenize=False, pos="v"):
    """
    `words`: a string or a list of strings
    -- set `not_tokenized` to False if this is a list of strings.
    If `not_tokenized` is True, the string `words` will be split into tokens by spaces.
    Set `do_not_tokenize` if `words` is not tokenized but shouldn't be split
    (useful if `words` is a single word).

    `pos` is a part-of-speech code accepted by nltk.

    Returns a list of lemmatized words.
    """
    from nltk.stem import WordNetLemmatizer

    if not_tokenized:
        to_lemm = words.split()
    else:
        to_lemm = words

    lemmatizer = WordNetLemmatizer()

    if do_not_tokenize:
        return [lemmatizer.lemmatize(words, pos=pos)]

    return [lemmatizer.lemmatize(token, pos=pos) for token in to_lemm]


def stem_string(words, not_tokenized=True, do_not_tokenize=False, language="english"):
    """
    `words` should be a string or a list of strings; set `not_tokenized` to False if this is a list of strings.
    If `not_tokenized` is True, the string `words` will be split into tokens by spaces.
    Set `do_not_tokenize` if `words` is not tokenized but shouldn't be split (useful if `words` is a single word).

    `language` is a language accepted by nltk.

    Returns a list of stemmed words.
    """
    from nltk.stem import SnowballStemmer

    if not_tokenized:
        to_stem = words.split()
    else:
        to_stem = words

    stemmer = SnowballStemmer(language=language)

    if do_not_tokenize:
        return [stemmer.stem(words)]

    return [stemmer.stem(token) for token in to_stem]


def fix_contractions(words):
    """
    `words` should be a string. Only English text is accepted.

    Returns a list of un-contracted words.
    """
    import contractions

    try:
        return contractions.fix(words)
    except IndexError:
        # On rare occasions, the contractions library crashes with special characters
        return words


def plot_data_quantities(
    df,
    col,
    days_interval,
    start_date=None,
    end_date=None,
    normalize=False,
    show_plot=True,
    color=None,
    plot_title="Quantity of data in time frames",
    hide_xticks=False,
):
    """
    Makes a matplot graph of of the numbers of posts over time. Requires a `col` where
    the timestamps are stored, a `data_format` to allow `datetime` to parse the timestamp,
    and a `days_interval` to tell how large each time interval is.
    Plots data between `start_date` and `end_date`, defaulting to the earliest and latest
    date in the data. Pass `hide_xticks` to suppress the ticks/labels.
    You can choose to automatically display the
    generated plot or not with the `show_plot` flag. Returns the x and y axes.
    The `color` argument must be `None` or a valid matplotlib color code
    """

    import matplotlib.pyplot as plt
    from pandas import to_datetime, Timestamp, DataFrame

    dt_series = to_datetime(df[col])

    if start_date is None:
        beginning = dt_series.min()
    else:
        beginning = Timestamp(start_date)

    if end_date is None:
        end = dt_series.max()
    else:
        end = Timestamp(end_date)

    # Construct timestamp-oriented DataFrame
    ts_df = DataFrame({"ts": dt_series.tolist(), "date": df[col].tolist()})
    ts_df = ts_df[(ts_df["ts"] >= beginning) & (ts_df["ts"] <= end)]
    ts_df = (
        ts_df.set_index("ts")
        .resample(str(days_interval) + "D")
        .agg({"date": "count"})
        .reset_index()
    )

    # Construct axes
    x_axis_labels = [x.strftime("%Y-%m-%d") for x in ts_df["ts"]]
    y_axis_quantities = [y / ts_df["date"].sum() if normalize else y for y in ts_df["date"]]

    # Construct plot
    plt.bar(x_axis_labels, y_axis_quantities, color=color)
    p_title = plot_title
    if normalize:
        p_title += " (data normalized)"
        plt.ylabel("Fraction of total documents")
    else:
        plt.ylabel("Number of documents")

    plt.title(p_title)
    plt.xlabel("Start day of time frame")
    if hide_xticks:
        plt.xticks(visible=False)
    else:
        plt.xticks(rotation=45)

    if show_plot:
        plt.show()

    return x_axis_labels, y_axis_quantities
