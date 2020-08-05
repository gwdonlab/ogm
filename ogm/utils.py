"""
Commonly-used utility functions
This is mainly here to reduce code duplication
"""

# pylint: disable=too-many-branches


def text_data_preprocess(setup_dict, output=True):
    """
    Run preprocessing on text data; `setup_dict` should be in JSON format described in README.
    Set `output` to write processed data into a file, otherwise this will return the parser's data.
    """
    from ogm.parser import TextParser
    import re

    # Absolute path to data file
    data_file = setup_dict["input_path"]

    # Optional language attribute in setup
    if "lang" in setup_dict:
        parser = TextParser(language=setup_dict["lang"])
    else:
        parser = TextParser()

    # Optional text encoding attribute in setup
    if "encoding" in setup_dict:
        file_encoding = setup_dict["encoding"]
    else:
        file_encoding = "iso8859"

    if data_file.endswith(".tsv"):
        parser.parse_csv(data_file, delimiter="\t", encoding=file_encoding)
    elif data_file.endswith(".csv"):
        parser.parse_csv(data_file, encoding=file_encoding)
    elif data_file.endswith(".xlsx"):
        parser.parse_excel(data_file)
    elif data_file.endswith(".rds"):
        parser.parse_rds(data_file)
    else:
        raise IOError("Unsupported file type")

    # Optional time frame attribute
    if "time_filter" in setup_dict:
        parser.filter_within_time_range(
            setup_dict["time_filter"]["time_key"],
            setup_dict["time_filter"]["data_format"],
            setup_dict["time_filter"]["arg_format"],
            setup_dict["time_filter"]["start"],
            setup_dict["time_filter"]["end"],
        )

    # Other attribute filters
    if "attribute_filters" in setup_dict:
        for attr_filter in setup_dict["attribute_filters"]:
            parser.filter_data(
                attr_filter["filter_key"], set(attr_filter["filter_vals"])
            )

    if "remove_emoji" in setup_dict:
        # Taken from this Gist:
        # https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b#gistcomment-3315605

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

        for item in parser.data:
            item[setup_dict["text_key"]] = emoji_pattern.sub(
                r" ", item[setup_dict["text_key"]]
            )

    if "replace_before_stemming" in setup_dict:
        parser.replace_words(
            setup_dict["text_key"], setup_dict["replace_before_stemming"]
        )

    if "remove_before_stemming" in setup_dict:
        parser.remove_words(
            setup_dict["text_key"], set(setup_dict["remove_before_stemming"])
        )

    parser.lemmatize_stem_words(setup_dict["text_key"])

    if "replace_after_stemming" in setup_dict:
        parser.replace_words(
            setup_dict["text_key"], setup_dict["replace_after_stemming"]
        )

    if "remove_after_stemming" in setup_dict:
        parser.remove_words(
            setup_dict["text_key"], set(setup_dict["remove_after_stemming"])
        )

    if "min_length" in setup_dict:
        parser.filter_doc_length(setup_dict["text_key"], setup_dict["min_length"])

    # Required output path attribute if this arg is true
    if output:
        parser.write_csv(
            setup_dict["output_path"], delimiter="\t", encoding=file_encoding
        )

        return None

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
