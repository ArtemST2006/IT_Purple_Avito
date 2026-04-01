from pydantic import BaseModel
from typing import List

class KeyPhrases(BaseModel):
    mcTitle: str
    keyPhrases: List[str]
    description: str

