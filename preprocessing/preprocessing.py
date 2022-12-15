from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from .dictionaries import emoticons, simple_contractions, haha
from ekphrasis.classes.preprocessor import TextPreProcessor
import re
import os.path as path
import os
import numpy as np
import shutil
import docker
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)


def apply_S2S(dataset_file):
    """Preprocess a dataset using seq2seq model. 
    WARNING: Uses docker running on the local machine. Make sure docker is up and running.

    Args:
        dataset_file (bool): Path to the dataset file

    Returns:
        Path: Path to the processed file
    """
    # copy input file to be processed
    dir = os.path.dirname(os.path.realpath(__file__))
    shutil.copy(dataset_file, dir+"/seq2seq/raw_input/data.txt")

    client = docker.from_env()

    # build docker container if not yet present
    if not 'cil-maml-s2s:latest' in _flatten([image.tags for image in client.images.list()]):
        print("building docker container")
        client.images.build(path=dir+"/seq2seq", tag="cil-maml-s2s")

    # run model in docker container
    print("running model")
    client.containers.run(
        image='cil-maml-s2s', volumes={dir+"/seq2seq": {'bind': '/app', 'mode': 'rw'}})

    # clean up files after processing is done
    os.remove(dir+"/seq2seq/raw_input/data.txt")
    os.remove(dir+"/seq2seq/json_input/data.json")
    for f in os.listdir(dir+"/seq2seq/json_result"):
        os.remove(os.path.join(dir+"/seq2seq/json_result", f))

    return Path(dir+"/seq2seq/raw_result/result.txt")


def _flatten(xss):
    return [x for xs in xss for x in xs]


def sentence_statistics_from_file(dataset_file):
    with open(dataset_file, mode='r') as in_file:
        return sentence_statistics(in_file)

def sentence_statistics(sentences):
    """Count occurrences of token types that might be indicative of the sentiment in a sentence.

    Args:
        sentences (iterable): Iterable containing sentences to process

    Returns:
        list: statistics per line in the following format:
            [
                ...
                [
                    (num) hashtags
                    (num) duplicate_chars
                    (num) slang
                    (num) american english replacements
                    (num) contractions
                    (num) haha
                    (num) emoticons
                    (num) stopwords
                    (num) numbers
                    (num) numbers_and_letters
                ],
                ...
            ]
    """
    noslang_dict = _load_noslang_data()
    ea_dict = _load_ea_data()
    tokenizer = _build_tokenizer()

    sentence_statistics = []

    for line in sentences:
        statistics = {
            'hashtags': 0,
            'duplicate_chars': 0,
            'slang': 0,
            'ea': 0,
            'contractions': 0,
            'haha': 0,
            'emoticons': 0,
            'stopwords': 0,
            'numbers': 0,
            'numbers_and_letters': 0,
        }

        statistics["hashtags"] += len(re.findall(r"#\S", line))
        statistics["duplicate_chars"] += len(
            re.findall(r"([a-z])\1{2,}", line))
        for regex, _ in simple_contractions.items():
            statistics["contractions"] += len(re.findall(regex, line))
        statistics["haha"] += len(list(re.findall(haha, line)))

        tokens = tokenizer(line)

        for i, t in enumerate(tokens):
            if t in noslang_dict:
                statistics["slang"] += 1
            if t in ea_dict:
                statistics["ea"] += 1
            if t in emoticons:
                statistics["emoticons"] += 1
            if t in stopwords.words('english'):
                statistics["stopwords"] += 1

        statistics["numbers"] += len(re.findall(r"\b[0-9.]+\b", line))
        line = re.sub(r"\b[0-9.]+\b", "<NUMBER>", line)
        statistics["numbers_and_letters"] += len(
            re.findall(r"\b[\w0-9]*[0-9]+[\w0-9]*\b", line))

        sentence_statistics.append(list(statistics.values()))

    return sentence_statistics


def preprocess_data(dataset_file,
                    unpack_hashtags=False,
                    remove_duplicate_chars=False,
                    removal_contractions=False,
                    simplify_haha=False,
                    dict_methods=False,
                    replace_emoticons=False,
                    remove_stopwords=False,
                    lemmatize=False,
                    stemming=False,
                    replace_numbers=False,
                    s2s=False,
                    **kwargs):
    """Preprocess a dataset.

    Results will be cached to the local file system. While processing the file, the function counts
    the number of replacements on the word level (how many words have been replaced per preprocessing option)
    and the sentence level (how many sentences had at least one replacement of a certain type).

    Args:
        dataset_file (string): Path to the dataset file
        unpack_hashtags (bool): Splits hashtags into seperate words if possible: #iloveyou - i love you,
        remove_duplicate_chars (bool): Removes consecutive duplicate chars from words: yaaay - yaay,
        removal_contractions (bool): Replaces standard english contractions: isn't - is not
        simplify_haha (bool): Unifies different versions of the expression haha: hahahahha - haha
        dict_methods (bool): Replaces slang and non-standard english tokens: lol - laughing out loud
        replace_emoticons (bool): Replaces emoticons with special tokens representing the sentiment: 
        remove_stopwords (bool): Removes stopwords using nltk stopwords
        lemmatize (bool): Lemmatize, by default using nltk WordNetLemmatizer()
        stemming (bool): Stem words, by default using the nltk SnowballStemmer('english')
        replace_numbers (bool): replaces numbers with special token
        s2s (bool): applies seq to seq model for normalization. WARN: uses docker on local machines

    Kwargs:
        tokenizer (Function): takes a sentence and splits it into seperate tokens
        lemmatizer (Function): takes list of tokens (strings) and returns processed tokens
        stemmer (Function): takes list of tokens (strings) and returns processed tokens

    Returns:
        (Path, dict, dict): Path to the preprocessed file, dict of word-level statistics, dict of sentence-level statistics
    """

    _, *bool_args, dependencies = locals().values()
    cache_name = _get_cache_name(dataset_file, bool_args)
    cache_file = Path(path.dirname(
        path.abspath(__file__)) + "/cache/" + cache_name)

    if cache_file.exists():
        return cache_file, {}, {}
    elif not s2s:
        return _preprocess_data(dataset_file, cache_file, bool_args, dependencies)
    else:
        return _preprocess_data_with_s2s(dataset_file, cache_file, bool_args, dependencies)


def _get_cache_name(dataset_file, bool_args):
    argument_mask = ''.join([str(val)[0] for val in bool_args])
    path = Path(dataset_file)
    return argument_mask + path.name


def _preprocess_data(dataset_file,
                     cache_file,
                     bool_args,
                     dependencies):

    word_statistics = {
        'hashtags': 0,
        'duplicate_chars': 0,
        'slang': 0,
        'ea': 0,
        'contractions': 0,
        'haha': 0,
        'emoticons': 0,
        'stopwords': 0,
        'numbers': 0,
        'numbers_and_letters': 0,
    }

    sentence_statistics = {
        'hashtags': 0,
        'duplicate_chars': 0,
        'slang': 0,
        'ea': 0,
        'contractions': 0,
        'haha': 0,
        'emoticons': 0,
        'stopwords': 0,
        'numbers': 0,
        'numbers_and_letters': 0,
    }

    word_statistics_counts = np.array(list(word_statistics.values()))
    sentence_statistics_counts = np.array(list(sentence_statistics.values()))

    noslang_dict = _load_noslang_data()
    ea_dict = _load_ea_data()

    tokenizer = _build_tokenizer()
    if "tokenizer" in dependencies:
        tokenizer = dependencies["tokenizer"]

    lemmatizer = _build_lemmatizer()
    if "lemmatizer" in dependencies:
        lemmatizer = dependencies["lemmatizer"]

    stemmer = _build_stemmer()
    if "stemmer" in dependencies:
        stemmer = dependencies["stemmer"]

    hashtag_processor = TextPreProcessor(
        segmenter="twitter", unpack_hashtags=True)

    with open(dataset_file, mode='r') as in_file:
        with cache_file.open(mode='w') as out_file:
            for line in in_file:
                output, statistics = _process_line(
                    line, noslang_dict, ea_dict, tokenizer, hashtag_processor, lemmatizer, stemmer, *bool_args)
                out_file.write(output)

                stats = np.array(list(statistics.values()))
                word_statistics_counts += stats
                sentence_statistics_counts += np.sign(stats)

    word_statistics = dict(
        zip(word_statistics.keys(), list(word_statistics_counts)))
    sentence_statistics = dict(
        zip(sentence_statistics.keys(), list(sentence_statistics_counts)))

    return cache_file, word_statistics, sentence_statistics


def _process_line(line,
                  noslang_dict,
                  ea_dict,
                  tokenizer,
                  hashtag_processor,
                  lemmatizer,
                  stemmer,
                  unpack_hashtags=False,
                  remove_duplicate_chars=False,
                  removal_contractions=False,
                  simplify_haha=False,
                  dict_methods=False,
                  replace_emoticons=False,
                  remove_stopwords=False,
                  lemmatize=False,
                  stemming=False,
                  replace_numbers=False,
                  s2s=False):

    statistics = {
        'hashtags': 0,
        'duplicate_chars': 0,
        'slang': 0,
        'ea': 0,
        'contractions': 0,
        'haha': 0,
        'emoticons': 0,
        'stopwords': 0,
        'numbers': 0,
        'numbers_and_letters': 0,
    }

    if unpack_hashtags:
        statistics["hashtags"] += len(re.findall(r"#\S", line))
        line = hashtag_processor.pre_process_doc(line) + "\n"

    if remove_duplicate_chars:
        statistics["duplicate_chars"] += len(
            re.findall(r"([a-z])\1{2,}", line))
        line = _remove_consec_duplicates(line)

    if removal_contractions:
        for regex, replacement in simple_contractions.items():
            statistics["contractions"] += len(re.findall(regex, line))
            line = regex.sub(replacement, line)

    if simplify_haha:
        statistics["haha"] += len(list(filter(lambda s: s !=
                                  "haha", re.findall(haha, line))))
        line = _simplify_haha_in_line(line)

    tokens = tokenizer(line)

    if dict_methods:
        for i, t in enumerate(tokens):
            if t in noslang_dict:
                statistics["slang"] += 1
                tokens[i] = noslang_dict[t]
            if t in ea_dict:
                statistics["ea"] += 1
                tokens[i] = ea_dict[t]

    if replace_emoticons:
        for i, t in enumerate(tokens):
            if t in emoticons:
                statistics["emoticons"] += 1
                tokens[i] = emoticons[t]

    if remove_stopwords:
        new_tokens = []
        for i, t in enumerate(tokens):
            if t in stopwords.words('english'):
                statistics["stopwords"] += 1
            else:
                new_tokens.append(t)
        tokens = new_tokens

    if lemmatize:
        tokens = lemmatizer(tokens)

    if stemming:
        tokens = stemmer(tokens)

    line = " ".join(tokens) + "\n"

    if replace_numbers:
        statistics["numbers"] += len(re.findall(r"\b[0-9.]+\b", line))
        line = re.sub(r"\b[0-9.]+\b", "<NUMBER>", line)
        statistics["numbers_and_letters"] += len(
            re.findall(r"\b[\w0-9]*[0-9]+[\w0-9]*\b", line))
        line = re.sub(r"\b[\w0-9]*[0-9]+[\w0-9]*\b",
                      "<NUMBER_AND_LETTERS>", line)

    # run again in case spell checker has replaced a few things
    if removal_contractions:
        for regex, replacement in simple_contractions.items():
            statistics["contractions"] += len(re.findall(regex, line))
            line = regex.sub(replacement, line)

    return line, statistics


def _build_tokenizer():
    tokenizer = WhitespaceTokenizer()

    def tokenize(line):
        return tokenizer.tokenize(line)
    return tokenize


def _build_lemmatizer():
    wn_lemmatizer = WordNetLemmatizer()

    def lemmatize(tokens):
        word_pos_tags = nltk.pos_tag(tokens)
        return [wn_lemmatizer.lemmatize(tag[0], _get_wordnet_pos(tag[1]))
                for idx, tag in enumerate(word_pos_tags)]
    return lemmatize


def _build_stemmer():
    stemmer = SnowballStemmer('english')

    def stem(tokens):
        return [stemmer.stem(token) for token in tokens]
    return stem


def _get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def _simplify_haha_in_line(line):
    return re.sub(haha, "haha", line)


def _remove_consec_duplicates(string):
    new_string = ""
    one_back = ""
    two_back = ""
    for char in string:
        if len(new_string) < 2:
            new_string += char
            two_back = one_back
            one_back = char
        elif char.isalpha() and char == one_back and one_back == two_back:
            continue
        else:
            new_string += char
            two_back = one_back
            one_back = char
    return new_string


def _load_noslang_data():
    noslang_dict = {}
    dir = os.path.dirname(os.path.realpath(__file__))
    infile = open(dir+"/noslang_mod.txt", 'r')
    for line in infile:
        items = line.split(' - ')
        if len(items[0]) > 0 and len(items) > 1:
            noslang_dict[items[0].strip()] = items[1].strip()
    return noslang_dict


def _load_ea_data():
    ea_dict = {}
    dir = os.path.dirname(os.path.realpath(__file__))
    english = open(dir + "/englishspellings.txt", 'r')
    american = open(dir + "/americanspellings.txt", 'r')
    for line in english:
        ea_dict[line.strip()] = american.readline().strip()
    return ea_dict


def _preprocess_data_with_s2s(dataset_file,
                              cache_file,
                              bool_args,
                              kwargs):
    print("applying first part of preprocessing")
    print(bool_args)
    args1 = [*bool_args[0:6], False, False, False, False, False]
    first_step_result, word_stats_1, sentence_stats_1 = _preprocess_data(dataset_file,
                                                                         cache_file,
                                                                         args1,
                                                                         kwargs)
    print("applying S2S")
    s2s_result = apply_S2S(os.path.abspath(first_step_result))
    print("applying last part of preprocessing")
    args2 = [False, False, False, False, False, False, *bool_args[6:10], False]
    second_step_result, word_stats_2, sentence_stats_2 = _preprocess_data(os.path.abspath(s2s_result),
                                                                          cache_file,
                                                                          args2,
                                                                          kwargs)

    os.remove(s2s_result)
    return second_step_result, dict(Counter(word_stats_1) + Counter(word_stats_2)), dict(Counter(sentence_stats_1) + Counter(sentence_stats_2))
