import uuid
from filterCascade import FilterCascade


class PaddedCascade(FilterCascade):
    def __init__(
        self,
        positives,
        negatives,
        targetpos,
        targetneg,
        fprs=None,
        multi_process=False,
        margin=1.05,
    ):
        assert len(positives) <= targetpos
        assert len(negatives) <= targetneg
        # pad the two sets of UUIDs by adding more random UUIDs until targets are reached
        padpos = [uuid.uuid4() for _ in range(targetpos - len(positives))]
        padpos += [u for u in positives]
        padneg = [uuid.uuid4() for _ in range(targetneg - len(negatives))]
        padneg += [u for u in negatives]
        super().__init__(padpos, padneg, fprs, multi_process, margin)
