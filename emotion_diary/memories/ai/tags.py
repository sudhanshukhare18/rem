
from collections import Counter
from typing import List
from .loader import get_nlp

ALLOWED_POS={
    "NOUN",
    "PROPN",
    "ADJ"
}

def extract_tags(
        text:str,
        limit:int=10,
) -> List[str]:
    if not text or not text.strip():
        return []
    
    nlp = get_nlp()

    doc=nlp(text.lower())

    keyword=[]

    for Token in doc:
        if(Token.is_stop
           or Token.is_punct
           or Token.is_space
           or Token.like_num
        ):
            continue
        if Token.pos_ not in ALLOWED_POS:
            continue
        lemma = Token.lemma_.strip()

        if len(lemma)<3:
            continue

        keyword.append(lemma)
    counts = Counter(keyword)

    return [
        word
        for word , _ in counts.most_common(limit)
    ]

            
