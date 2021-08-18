# Online Group Modeling
Package for using machine learning to analyze the behavior of online groups

## Installation

To install all dependencies, you'll need to run these commands:

```bash
pip install -r requirements.txt
```

Finally, run:
```bash
python setup.py install
```

## Text Preprocessing Utility

Here is an example of a JSON-structure which could be used for the `text_data_preprocess` function in `ogm.utils`. Only `text_key` and `data_path` are always required, and `output_path` is required if `output` is true.

```json
{
    "text_key": "heading in the data table corresponding to the text of the posts",
    "data_path": "path to data table",
    "encoding": "encoding of the data table file, defaults to UTF-8",
    "lang": "language code of text data, defaults to 'en'",
    "attribute_filters": [
        {
            "filter_key": "data table heading",
            "filter_vals": [
                "list",
                "of",
                "values",
                "to",
                "include",
            ]
        }
    ],
    "time_filter": {
        "arg_format": "Python datetime formatting code for entries in this file",
        "start": "timestamp formatted according to arg_format",
        "end": "timestamp formatted according to arg_format",
        "data_format": "Python datetime formatting code for entries in the data",
        "time_key" : "heading in data table where timestamps appear"
    },
    "replace_before_stemming" : {
        "replace this": "with this",
        "and this": "with this"
    },
    "replace_after_stemming": {
        "same structure": "as above"
    },
    "remove_before_stemming": [
        "strings",
        "to",
        "remove"
    ],
    "remove_after_stemming": [
        "strings",
        "to",
        "remove"
    ],
    "remove_emoji" : "boolean",
    "output_path": "path to output file"
}
```

To try it out, run [`examples/text_test.py`](examples/text_test.py). It loads a config file that directs the preprocessor to remove the words "grandmother" and "grandfather" from six opening lines of notable works of literature. 