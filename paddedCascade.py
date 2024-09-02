import uuid
from filterCascade import FilterCascade


class PaddedCascade(FilterCascade):
    def __init__(self, positives, negatives, numpos, numneg, fprs=None):
        # pad the two sets of UUIDs by adding more random UUIDs until positives has numpos entries and negatives has numneg entries
        padpos = [uuid.uuid4() for _ in range(numpos - len(positives))]
        padpos += [u for u in positives]
        padneg = [uuid.uuid4() for _ in range(numneg - len(negatives))]
        padneg += [u for u in negatives]
        super().__init__(padpos, padneg, fprs)
