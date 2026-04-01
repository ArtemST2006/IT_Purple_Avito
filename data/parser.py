import csv
from typing import Dict

from src.schema import KeyPhrases


def parseCSV(path: str) -> Dict[int, KeyPhrases]:
    key_phrases_data: Dict[int, KeyPhrases] = {}
    try:
        with open(path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for idx, row in enumerate(reader):
                phrases_src = [p.strip() for p in row['keyPhrases'].split(';') if p.strip()]

                model_instance_src = KeyPhrases(
                    mcTitle=row['mcTitle'],
                    keyPhrases=phrases_src,
                    description=row['description']
                )

                key_phrases_data[idx] = model_instance_src

        print(f"Загружено записей: {len(key_phrases_data)}")
        return key_phrases_data

    except Exception as e:
        print(f"Ошибка при чтении CSV: {e}")
        return {}