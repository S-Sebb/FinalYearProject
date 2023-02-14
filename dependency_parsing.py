# -*- coding: utf-8 -*-
import stanza
from stanza import DownloadMethod


def dependency_parsing(text):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', logging_level="ERROR")
    doc = nlp(text)
    extended_annotations = []
    for sent in doc.sentences:
        for word in sent.words:
            extended_annotations.append(word)
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    return extended_annotations
