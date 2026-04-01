import re
import pymorphy3
import torch

from sentence_transformers import SentenceTransformer, util
from typing import Dict, List
from src.schema import KeyPhrases

class DetectorProcess:
    COSINE_COEFFICIENT = 0.5
    ALLOWED_POS = {'NOUN', 'ADJF', 'ADJS', 'INFN'}

    key_phrases_data: Dict[int, KeyPhrases] = {}
    key_phrases_data_lem: Dict[int, KeyPhrases] = {}
    key_phrases_embeddings: Dict[int, torch.Tensor] = {}
    microcategories: Dict[int, str] = {}
    result_mCategories: List[int] = []

    __morph = pymorphy3.MorphAnalyzer()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def __init__(self, keys: Dict[int, KeyPhrases]):
        self.key_phrases_data = keys
        self.init_data()

    def init_data(self) -> None:
        for key, value in self.key_phrases_data.items():
            phrases_lem = [self.__lemmatization(p.strip().lower()) for p in value.keyPhrases if p.strip()]
            model_instance_lem = KeyPhrases(
                mcTitle=value.mcTitle,
                keyPhrases=phrases_lem,
                description=value.description
            )

            combined_text = f"{value.mcTitle} " + " ".join(value.keyPhrases)
            # combined_text = self.__lemmatization(value.mcTitle)
            embedding = self.model.encode(combined_text, convert_to_tensor=True)

            self.key_phrases_data_lem[key] = model_instance_lem
            self.microcategories[key] = value.mcTitle
            self.key_phrases_embeddings[key] = embedding


    def detect(self, text: str) -> None:
        detected_by_keywords = self.lemmatization(text)
        detected_by_semantics = self.cosine_proximity(text)
        self.result_mCategories = list(set(detected_by_keywords) | set(detected_by_semantics))

    def lemmatization(self, text: str) -> List[int]:
        if not text:
            return []
        target_lemmas = self.__lemmatization(text).split()
        target_set = set(target_lemmas)

        matched_category_ids = []

        for mc_id, key_phrases_model in self.key_phrases_data_lem.items():
            for phrase_str in key_phrases_model.keyPhrases:
                phrase_words = phrase_str.split()
                phrase_set = set(phrase_words)

                if phrase_set.issubset(target_set):
                    if self.__is_close(phrase_words, target_lemmas):
                        matched_category_ids.append(mc_id)
                        break

        return matched_category_ids

    def __lemmatization(self, text: str) -> str:
        text = text.lower()

        words = re.findall(r'[а-яёa-z]+', text)

        res = []
        for word in words:
            p = self.__morph.parse(word)[0]
            res.append(p.normal_form)
        return " ".join(res)

    def __is_close(self, phrase_words, target_lemmas, window=5):
        """
        Проверяет, что слова из фразы находятся не слишком далеко друг от друга.
        window=5 означает, что между словами фразы не более 5 других слов.
        """
        if len(phrase_words) < 2: return True

        indices = [i for i, x in enumerate(target_lemmas) if x in phrase_words]
        if not indices: return False

        return (max(indices) - min(indices)) <= (len(phrase_words) + window)


    def cosine_proximity(self, text: str, coef: float = COSINE_COEFFICIENT) -> List[int]:
        # text = self.__lemmatization(text)
        chunks = self.__get_semantic_chunks(text)
        if not chunks:
            return []

        clean_chunks = []
        for chunk in chunks:
            pos_cleaned = self.__apply_pos_filter(chunk)
            clean_chunks.append(pos_cleaned)

        if not clean_chunks:
            return []

        chunk_embeddings = self.model.encode(clean_chunks, convert_to_tensor=True)
        detected_mc_ids = set()

        for mc_id, cat_embedding in self.key_phrases_embeddings.items():
            cosine_scores = util.cos_sim(chunk_embeddings, cat_embedding)
            if torch.max(cosine_scores) > coef:
                detected_mc_ids.add(mc_id)

        return list(detected_mc_ids)

    def __apply_pos_filter(self, chunk: str) -> str:
        """
        Оставляет в чанке только слова разрешенных частей речи (сущ, прил, инфинитивы).
        Это убирает "шумные" глаголы, союзы и наречия.
        """
        words = re.findall(r'[а-яёa-z]+', chunk.lower())
        filtered_words = []

        for word in words:
            p = self.__morph.parse(word)[0]
            if p.tag.POS in self.ALLOWED_POS:
                filtered_words.append(word)

        return " ".join(filtered_words)

    def __get_semantic_chunks(self, text: str) -> List[str]:
        """
        Разбивает текст на смысловые фрагменты по пунктуации и союзам.
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        split_pattern = r'[,.;!?\n]|\b(?:и|или|а также|также|плюс|включая|в том числе)\b'

        raw_chunks = re.split(split_pattern, text)

        valid_chunks = []
        for chunk in raw_chunks:
            cleaned_chunk = chunk.strip()
            if len(cleaned_chunk.split()) >= 1:
                valid_chunks.append(cleaned_chunk)

        return valid_chunks


    def get_ides(self) -> List[int]:
        return self.result_mCategories

    def get_titles(self) -> List[str]:
        return [self.microcategories[i] for i in self.result_mCategories]

    def __str__(self):
        s = ""
        for key, value in self.key_phrases_data_lem.items():
            s += value.mcTitle + "\n"

        return s

