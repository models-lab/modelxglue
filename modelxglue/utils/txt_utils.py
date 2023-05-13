from re import finditer


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def tokenizer(doc, separator='\n'):
    words = doc.split(separator)
    # split _
    words = [w2 for w1 in words for w2 in w1.split('_') if w2 != '']
    # camelcase
    words = [w2.lower() for w1 in words for w2 in camel_case_split(w1) if w2 != '']
    return words
