"""
Commonly-used utility functions
This is mainly here to reduce code duplication
"""


def text_data_preprocess(setup_dict, output=True):
    """
    Run preprocessing on text data; `setup_dict` should be in JSON format described in README.
    Set `output` to write processed data into a file, otherwise this will return the parser's data.
    """
    from ogm.parser import TextParser

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

    else:
        return parser.data
