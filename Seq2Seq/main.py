import spacy
import torch
import torchtext


spacy_ru = spacy.load('ru')
spacy_en = spacy.load('en')


def tokenizer_ru(text):
    return [tok.text for tok in spacy_ru.tokenizer(text)]


def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

