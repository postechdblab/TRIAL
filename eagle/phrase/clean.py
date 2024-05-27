import itertools
import re
import sys
import unicodedata
from typing import *

import hkkang_utils.file as file_utils
import torch
from unidecode import unidecode

all_chars = (chr(i) for i in range(sys.maxunicode))
categories = {"Cc"}
control_chars = "".join(c for c in all_chars if unicodedata.category(c) in categories)
# or equivalently and much more efficiently
control_chars = "".join(map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0))))
control_char_re = re.compile("[%s]" % re.escape(control_chars))

pattern1 = re.compile(r"\[PAD\]")
pattern2 = re.compile(r"\x10")


def remove_control_chars(s):
    return control_char_re.sub("", s)


def replace_pad_with_pad(text: str) -> str:
    # Use the compiled pattern to replace all occurrences of "[PAD]" with "[pad]".
    return pattern1.sub("[pad]", text)


def remove_x10_character(text: str) -> str:
    # Use the compiled pattern to replace all occurrences of "\x10" with "".
    return pattern2.sub("", text)


def remove_double_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def unidecode_text(text: str, rm_control_chars: bool = True) -> str:
    text = remove_double_spaces(text)
    text = replace_pad_with_pad(text)
    if rm_control_chars:
        text = remove_control_chars(text)
    return unidecode(text)
