!pip install autocorrect -q

from nltk import download
from nltk.tokenize import TweetTokenizer
download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.sentiment.util import mark_negation

lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords
import sys
import re
import nltk.data
from nltk import pos_tag_sents
# If you have Anaconda, you can install emoji using
# "conda install -c conda-forge emoji" command. You can download autocorrect using pip
# and "target" parameter: "pip install autocorrect --target=<directory>"
from autocorrect import Speller
import emoji
# from emoji import get_emoji_regexp
# replaced at version 2.00 https://carpedm20.github.io/emoji/docs/
# if you want to use regex:
# def get_emoji_regexp():  If you want to use regex
#     # Sort emoji by length to make sure multi-character emojis are
#     # matched first
#     emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
#     pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
#     return re.compile(pattern)
# get_emoji_regexp = get_emoji_regexp()


# Note that it looks like the POS tagger prefers us to feed sentences separately (or
# feed them as a list to pos_tag_sents()). Therefore, we will tokenize sentences first.
# Sentence tokenizer tokenizes sentences while also trying to handle periods that do not
# function as a sentence terminator (such as the period in "Mr.").
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# This function wraps up all the processes and returns normalized tweets (optionally with
# bigrams). It has some parameters you can play with. To keep it monolithic and easier to
# analyze, it is written as a one big function. From a software engineering perspective,
# it would make more sense to move certain parts to their own functions to separate
# different concerns (subtasks) such as emoji removal, punctutation removal, etc. The
# function also has tokenizer parameters that have default values, which make sure that
# these objects exist when it needs them. You can also include import statements and such
# in the function to make it more portable, or you could make it a module.

def get_lemmatizer_pos(pos):
    pos_start = pos[0] # Takes the first letter to simplify the POS tag
    if pos_start == "J":
        return wn.ADJ
    elif pos_start == "V":
        return wn.VERB
    elif pos_start == "R":
        return wn.ADV
    else:
        return wn.NOUN

def tokenize_normalize(tweet, sentence_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle'),
                       tokenizer=TweetTokenizer(preserve_case=False), return_bigrams=False,
                       tokenize_numbers=False, tokenize_users=False, tokenize_hashtags=False,
                       tokenize_urls=False, remove_emoji=True, remove_punct=True,
                       handle_negation=True, autocorrect=False):

    # This retrieves a list of stop words in English, which will be used to remove the
    # stop words:
    stop_words = stopwords.words("english")

    # These combined punctuations will be used to remove punctuations from tweets (it
    # is an extension to string.punctuation):
    punctuations = "!\"“”#$%&'‘’()*+,-./:;<=>?@[\]^_`{|}~‍"

    # We will use this function to correct typographic errors:
    if autocorrect and "autocorrect" in sys.modules:
        spell = Speller()

    # Separates tweets into sentences:
    tweet_sentences = sentence_tokenizer.tokenize(tweet)

    # Tokenization outputs are kept in separate lists for each sentence:
    tweet_sentences_tokens = [tokenizer.tokenize(sentence) for sentence in tweet_sentences]

    # POS tagging happens separately for each sentence before they are combined:
    tokens_pos = [pos_tag for pos_tags in pos_tag_sents(tweet_sentences_tokens) for pos_tag in pos_tags]

    # For each POS-tagged token, a lemma is obtained:
    lemmas = [lemmatizer.lemmatize(token[0], pos=get_lemmatizer_pos(token[1])) for token in tokens_pos]
#     print(lemmas)

    # Marks negations:
    if handle_negation:
        lemmas = mark_negation(lemmas)

    filtered_lemmas = []
    bigrams = []
    last_filtered_lemma_index = None
    last_filtered_lemma = None
    for lemma_index, lemma in enumerate(lemmas):

        # The amount of emojis has skyrocketed, and the way new emojis or their
        # varients are added technically complicates handling emojis. For example,
        # some emojis are formed by combining different emojis and a zero-width joiner
        # in between. Removing variation selectors such as hair/skin color and gender
        # for emojis since they cause noise and tokenization problems:
        if re.sub("[\\uFE00-\\uFE0F♂♀‍]+", "", lemma) == "":
            continue

        # Filters hashtags:
        if lemma.startswith("#"):
            if tokenize_hashtags:
                lemma = "<hashtag>"
            else:
                continue

        # Filters user handles:
        if lemma.startswith("@"):
            if tokenize_users:
                lemma = "<user>"
            else:
                continue

        # Filters stop words (considers negations):
        if lemma.replace("_NEG", "") in stop_words:
            continue

        # Filters the lemma by searching for "https://," "http://," or "www." using
        # regular expression. If one of them exists, they are not retrieved. Regular
        # expression may seem daunting at first. It is not mandatory, but you can check
        # tutorials like this: https://regexone.com/lesson/introduction_abcs
        if re.search("(https?:\/\/)|(www\.)", lemma):
            if tokenize_urls:
                lemma = "<url>"
            else:
                continue

        # Filters emojis using emeji package (considers negations):
        if remove_emoji and "emoji" in sys.modules:
            # lemma = get_emoji_regexp.sub(u'', lemma.replace("_NEG", ""))
            lemma = emoji.replace_emoji(lemma.replace("_NEG", ""),u'')

        # Filters punctuation (considers negations):
        if remove_punct and lemma.replace("_NEG", "").translate(lemma.maketrans('', '', punctuations)) == "":
            continue

        # Corrects typographic errors using autocorrect package (considers negations):
        if autocorrect and "autocorrect" in sys.modules and spell:
            if "_NEG" in lemma:
                # Removing "_NEG" and adding it back after autocorrection:
                lemma_autocorrected = spell(lemma.replace("_NEG", "")).join("_NEG")
            else:
                lemma_autocorrected = spell(lemma)

            if lemma != lemma_autocorrected:
#                 print(lemma,"autocorrected to",lemma_autocorrected) # Uncomment this line to print the corrections
                lemma = lemma_autocorrected

        # Tries to convert a number from string to float while also handling commas
        # and percentage signs. If the token is a number, it is transformed to "<number>"
        # token or not retrieved. If not, it silently ignores the exception and
        # continues.
        try:
            float(lemma.replace(",", "").replace("%", ""))
            if tokenize_numbers:
                lemma = "<number>"
            else:
                continue
        except:
            pass

        # If the lemma survives all these processes, it is appended to the list
        filtered_lemmas.append(lemma)

        # If returning bigrams is set to True, this part extracts the bigrams:
        if return_bigrams:
            # If there is a last filtered lemma, if its location in the sentences is
            # right before the current lemma, and if the current lemma is not a
            # punctuation:
            if last_filtered_lemma and last_filtered_lemma_index + 1 == lemma_index and\
            lemma.replace("_NEG", "").translate(lemma.maketrans('', '', punctuations)) != "":
                # The lemma group (bigram) is appended to the bigram list
                bigrams.append([last_filtered_lemma, lemma])

            last_filtered_lemma_index = lemma_index
            last_filtered_lemma = lemma

    if return_bigrams:
        # It returns filtered lemmas and bigrams together
        return (filtered_lemmas, bigrams)
    else:
        return filtered_lemmas