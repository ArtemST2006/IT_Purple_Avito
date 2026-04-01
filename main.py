import os
from typing import Dict

from src.detector import DetectorProcess
from data.parser import parseCSV
from src.schema import KeyPhrases

PATH_KEYS_PHRASES = "data/csv_files/rnc_mic_key_phrases.csv"

def main():
    keys: Dict[int, KeyPhrases] = parseCSV(PATH_KEYS_PHRASES)
    obj = DetectorProcess(keys)

    text = "Оклейка стен обоями в коттедж. Делаю поклейка стеклообоев, с гарантией. частный мастер с опытом 7 лет. цены адекватные. есть свободные даты."
    lsp_cos = obj.cosine_proximity(text)
    lsp_lem = obj.lemmatization(text)

    print("_______cos________", end="\n")
    for el in lsp_cos:
        print(keys[el].mcTitle)

    print()
    print("_______lem________", end="\n")
    for el in lsp_lem:
        print(keys[el].mcTitle)


if __name__ == "__main__":
    main()