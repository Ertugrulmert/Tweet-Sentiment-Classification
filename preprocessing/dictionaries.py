import re

haha = re.compile(r"\ba?h+a+\-?h+a+\-?[h+a+\-?]*\b")

#gotten from https://github.com/ian-beaver/pycontractions
simple_contractions = {
    re.compile(r"\bcan'?t\b", re.I | re.U): "cannot",
    re.compile(r"\bcan'?t'?ve\b", re.I | re.U): "cannot have",
    re.compile(r"\b'?cause\b", re.I | re.U): "because",
    re.compile(r"\bcould'?ve\b", re.I | re.U): "could have",
    re.compile(r"\bcouldn'?t\b", re.I | re.U): "could not",
    re.compile(r"\bcouldn'?t'?ve\b", re.I | re.U): "could not have",
    re.compile(r"\bdidn'?t\b", re.I | re.U): "did not",
    re.compile(r"\bdoesn'?t\b", re.I | re.U): "does not",
    re.compile(r"\bdon'?t\b", re.I | re.U): "do not",
    re.compile(r"\bdoin'?\b", re.I | re.U): "doing",
    re.compile(r"\bdunno\b", re.I | re.U): "do not know",
    re.compile(r"\bgimme'?\b", re.I | re.U): "give me",
    re.compile(r"\bgoin'?\b", re.I | re.U): "going",
    re.compile(r"\bgonna'?\b", re.I | re.U): "going to",
    re.compile(r"\bhadn'?t\b", re.I | re.U): "had not",
    re.compile(r"\bhadn'?t'?ve\b", re.I | re.U): "had not have",
    re.compile(r"\bhasn'?t\b", re.I | re.U): "has not",
    re.compile(r"\bhaven'?t\b", re.I | re.U): "have not",
    re.compile(r"\bhe'?d'?ve\b", re.I | re.U): "he would have",
    re.compile(r"\bhow'?d\b", re.I | re.U): "how did",
    re.compile(r"\bhow'?d'?y\b", re.I | re.U): "how do you",
    re.compile(r"\bhow'?ll\b", re.I | re.U): "how will",
    re.compile(r"\bi'?d'?ve\b", re.I | re.U): "i would have",
    # May replace the abbreviation "im" as in Instant Messenger.
    # If this abbreviation is in your data remove the "?"
    re.compile(r"\bi'?m\b", re.I | re.U): "i am",
    re.compile(r"\bi'?ve\b", re.I | re.U): "i have",
    re.compile(r"\bisn'?t\b", re.I | re.U): "is not",
    re.compile(r"\bit'?d'?ve\b", re.I | re.U): "it would have",
    re.compile(r"\bkinda\b", re.I | re.U): "kind of",
    re.compile(r"\blet'?s\b", re.I | re.U): "let us",
    re.compile(r"\bma'?am\b", re.I | re.U): "madam",
    re.compile(r"\bmayn'?t\b", re.I | re.U): "may not",
    re.compile(r"\bmight'?ve\b", re.I | re.U): "might have",
    re.compile(r"\bmightn'?t\b", re.I | re.U): "might not",
    re.compile(r"\bmightn'?t'?ve\b", re.I | re.U): "might not have",
    re.compile(r"\bmust'?ve\b", re.I | re.U): "must have",
    re.compile(r"\bmustn'?t\b", re.I | re.U): "must not",
    re.compile(r"\bmustn'?t'?ve\b", re.I | re.U): "must not have",
    re.compile(r"\bnothin'?\b", re.I | re.U): "nothing",
    re.compile(r"\bneedn'?t\b", re.I | re.U): "need not",
    re.compile(r"\bneedn'?t'?ve\b", re.I | re.U): "need not have",
    re.compile(r"\bo'?clock\b", re.I | re.U): "of the clock",
    re.compile(r"\boughta\b", re.I | re.U): "ought to",
    re.compile(r"\boughtn'?t\b", re.I | re.U): "ought not",
    re.compile(r"\boughtn'?t'?ve\b", re.I | re.U): "ought not have",
    re.compile(r"\bshan'?t\b", re.I | re.U): "shall not",
    re.compile(r"\bsha'?n'?t\b", re.I | re.U): "shall not",
    re.compile(r"\bshan'?t'?ve\b", re.I | re.U): "shall not have",
    re.compile(r"\bshe'?d'?ve\b", re.I | re.U): "she would have",
    re.compile(r"\bshould'?ve\b", re.I | re.U): "should have",
    re.compile(r"\bshouldn'?t\b", re.I | re.U): "should not",
    re.compile(r"\bshouldn'?t'?ve\b", re.I | re.U): "should not have",
    re.compile(r"\bso'?ve\b", re.I | re.U): "so have",
    re.compile(r"\bsomethin'?\b", re.I | re.U): "something",
    re.compile(r"\bthat'?d'?ve\b", re.I | re.U): "that would have",
    re.compile(r"\bthere'?d'?ve\b", re.I | re.U): "there would have",
    re.compile(r"\bthey'?d'?ve\b", re.I | re.U): "they would have",
    re.compile(r"\bthey'?re\b", re.I | re.U): "they are",
    re.compile(r"\bthey'?ve\b", re.I | re.U): "they have",
    re.compile(r"\b'?tis\b", re.I | re.U): "it is",
    re.compile(r"\bto'?ve\b", re.I | re.U): "to have",
    re.compile(r"\bu\b(?!\.)", re.I | re.U): "you",
    re.compile(r"\bwasn'?t\b", re.I | re.U): "was not",
    re.compile(r"\bwanna'?\b", re.I | re.U): "want to",
    re.compile(r"\bwe'?d'?ve\b", re.I | re.U): "we would have",
    re.compile(r"\bwe'll\b", re.I | re.U): "we will",
    re.compile(r"\bwe'?ll'?ve\b", re.I | re.U): "we will have",
    re.compile(r"\bwe're\b", re.I | re.U): "we are",
    re.compile(r"\bwe'?ve\b", re.I | re.U): "we have",
    re.compile(r"\bweren'?t\b", re.I | re.U): "were not",
    re.compile(r"\bwhat'?re\b", re.I | re.U): "what are",
    re.compile(r"\bwhat'?ve\b", re.I | re.U): "what have",
    re.compile(r"\bwhen'?ve\b", re.I | re.U): "when have",
    re.compile(r"\bwhere'?d\b", re.I | re.U): "where did",
    re.compile(r"\bwhere'?ve\b", re.I | re.U): "where have",
    re.compile(r"\bwho'?ve\b", re.I | re.U): "who have",
    re.compile(r"\bwhy'?ve\b", re.I | re.U): "why have",
    re.compile(r"\bwill'?ve\b", re.I | re.U): "will have",
    re.compile(r"\bwon'?t\b", re.I | re.U): "will not",
    re.compile(r"\bwon'?t'?ve\b", re.I | re.U): "will not have",
    re.compile(r"\bwould'?ve\b", re.I | re.U): "would have",
    re.compile(r"\bwouldn'?t\b", re.I | re.U): "would not",
    re.compile(r"\bwouldn'?t'?ve\b", re.I | re.U): "would not have",
    re.compile(r"\by'?all\b", re.I | re.U): "you all",
    re.compile(r"\by'?all'?d\b", re.I | re.U): "you all would",
    re.compile(r"\by'?all'?d'?ve\b", re.I | re.U): "you all would have",
    re.compile(r"\by'?all'?re\b", re.I | re.U): "you all are",
    re.compile(r"\by'?all'?ve\b", re.I | re.U): "you all have",
    re.compile(r"\byou'?d'?ve\b", re.I | re.U): "you would have",
    re.compile(r"\byou'?re\b", re.I | re.U): "you are",
    re.compile(r"\byou'?ve\b", re.I | re.U): "you have",
    re.compile(r"\baren'?t\b", re.I | re.U): "are not",
    re.compile(r"\bhe'll\b", re.I | re.U): "he will",
    re.compile(r"\bhe'?ll'?ve\b", re.I | re.U): "he will have",
    re.compile(r"\bI'll\b", re.I | re.U): "I will",
    re.compile(r"\bI'?ll'?ve\b", re.I | re.U): "I will have",
    re.compile(r"\bit'?ll\b", re.I | re.U): "it will",
    re.compile(r"\bit'?ll'?ve\b", re.I | re.U): "it will have",
    re.compile(r"\bshe'll\b", re.I | re.U): "she will",
    re.compile(r"\bshe'?ll'?ve\b", re.I | re.U): "she will have",
    re.compile(r"\bthey'?ll\b", re.I | re.U): "they will",
    re.compile(r"\bthey'?ll'?ve\b", re.I | re.U): "they will have",
    re.compile(r"\bwhat'?ll\b", re.I | re.U): "what will",
    re.compile(r"\bwhat'?ll'?ve\b", re.I | re.U): "what will have",
    re.compile(r"\bwho'?ll\b", re.I | re.U): "who will",
    re.compile(r"\bwho'?ll'?ve\b", re.I | re.U): "who will have",
    re.compile(r"\byou'?ll\b", re.I | re.U): "you will",
    re.compile(r"\byou'?ll'?ve\b", re.I | re.U): "you will have"
}

#adapted from https://github.com/cbaziotis/ekphrasis/blob/master/ekphrasis/dicts/emoticons.py
emoticons = {
    ':*': '<kiss>',
    ':-*': '<kiss>',
    ':x': '<kiss>',
    ':-)': '<happy>',
    ':-))': '<happy>',
    ':-)))': '<happy>',
    ':-))))': '<happy>',
    ':-)))))': '<happy>',
    ':-))))))': '<happy>',
    ':)': '<happy>',
    ':))': '<happy>',
    ':)))': '<happy>',
    ':))))': '<happy>',
    ':)))))': '<happy>',
    ':))))))': '<happy>',
    ':)))))))': '<happy>',
    ':o)': '<happy>',
    ':]': '<happy>',
    ':3': '<happy>',
    ':c)': '<happy>',
    ':>': '<happy>',
    '=]': '<happy>',
    '8)': '<happy>',
    '=)': '<happy>',
    ':}': '<happy>',
    ':^)': '<happy>',
    '|;-)': '<happy>',
    ":'-)": '<happy>',
    ":')": '<happy>',
    '\o/': '<happy>',
    '*\\0/*': '<happy>',
    ':-d': '<laugh>',
    ':d': '<laugh>',
    '8-d': '<laugh>',
    '8d': '<laugh>',
    'x-d': '<laugh>',
    'xd': '<laugh>',
    '=-d': '<laugh>',
    '=D': '<laugh>',
    '=-3': '<laugh>',
    '=3': '<laugh>',
    'b^d': '<laugh>',
    '>:[': '<sad>',
    ':-(': '<sad>',
    ':-((': '<sad>',
    ':-(((': '<sad>',
    ':-((((': '<sad>',
    ':-(((((': '<sad>',
    ':-((((((': '<sad>',
    ':-(((((((': '<sad>',
    ':(': '<sad>',
    ':((': '<sad>',
    ':(((': '<sad>',
    ':((((': '<sad>',
    ':(((((': '<sad>',
    ':((((((': '<sad>',
    ':(((((((': '<sad>',
    ':((((((((': '<sad>',
    ':-c': '<sad>',
    ':c': '<sad>',
    ':-<': '<sad>',
    ':<': '<sad>',
    ':-[': '<sad>',
    ':[': '<sad>',
    ':{': '<sad>',
    ':-||': '<sad>',
    ':@': '<sad>',
    ":'-(": '<sad>',
    ":'(": '<sad>',
    'd:<': '<sad>',
    'd:': '<sad>',
    'd8': '<sad>',
    'd;': '<sad>',
    'd=': '<sad>',
    'dX': '<sad>',
    'v.v': '<sad>',
    "d-':": '<sad>',
    '(>_<)': '<sad>',
    ':|': '<sad>',
    '>:O': '<surprise>',
    ':-O': '<surprise>',
    ':-o': '<surprise>',
    ':O': '<surprise>',
    '°o°': '<surprise>',
    'o_O': '<surprise>',
    'o_0': '<surprise>',
    'o.O': '<surprise>',
    'o-o': '<surprise>',
    '8-0': '<surprise>',
    '|-O': '<surprise>',
    ';-)': '<wink>',
    ';)': '<wink>',
    '*-)': '<wink>',
    '*)': '<wink>',
    ';-]': '<wink>',
    ';]': '<wink>',
    ';d': '<wink>',
    ';^)': '<wink>',
    ':-,': '<wink>',
    '>:p': '<tong>',
    ':-p': '<tong>',
    ':p': '<tong>',
    'x-': '<tong>',
    'x-p': '<tong>',
    'xp': '<tong>',
    ':-p': '<tong>',
    ':p': '<tong>',
    '=p': '<tong>',
    ':-Þ': '<tong>',
    ':Þ': '<tong>',
    ':-b': '<tong>',
    ':b': '<tong>',
    ':-&': '<tong>',
    '>:\\': '<annoyed>',
    '>:/': '<annoyed>',
    ':-/': '<annoyed>',
    ':-.': '<annoyed>',
    ':/': '<annoyed>',
    ':\\': '<annoyed>',
    '=/': '<annoyed>',
    '=\\': '<annoyed>',
    ':L': '<annoyed>',
    '=L': '<annoyed>',
    ':S': '<annoyed>',
    '>.<': '<annoyed>',
    ':-|': '<annoyed>',
    '<:-|': '<annoyed>',
    ':-x': '<seallips>',
    ':x': '<seallips>',
    ':-#': '<seallips>',
    ':#': '<seallips>',
    'o:-)': '<angel>',
    '0:-3': '<angel>',
    '0:3': '<angel>',
    '0:-)': '<angel>',
    '0:)': '<angel>',
    '0;^)': '<angel>',
    '>:)': '<devil>',
    '>:d': '<devil>',
    '>:-d': '<devil>',
    '>;)': '<devil>',
    '>:-)': '<devil>',
    '}:-)': '<devil>',
    '}:)': '<devil>',
    '3:-)': '<devil>',
    '3:)': '<devil>',
    'o/\o': '<highfive>',
    '^5': '<highfive>',
    '>_>^': '<highfive>',
    '^<_<': '<highfive>',
    '<3': '<heart>'
}
