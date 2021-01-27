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

Here is an example of a JSON-structure which could be used for the `text_data_preprocess` function in `ogm.utils`.

```json
{
    "text_key": "text",
    "replace_before_stemming": {
        "\n": " "
    },
    "remove_before_stemming": [
        "bit.ly"
    ],
    "remove_after_stemming": [
        "http",
        "www"
    ],
    "remove_emoji": true,
    "encoding": "utf8"
}
```