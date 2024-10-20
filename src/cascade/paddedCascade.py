from src.cascade.filterCascade import FilterCascade
import src.cascade.cascadeUtils as cu


class PaddedCascade(FilterCascade):
    def __init__(
        self,
        positives,
        negatives,
        targetpos,
        targetneg,
        fprs=None,
        k=None,
        multi_process=False,
    ):
        assert len(positives) <= targetpos
        assert len(negatives) <= targetneg
        # pad the two sets of UUIDs by adding more random UUIDs until targets are reached
        padpos = cu.gen_ids_wo_overlap(
            targetpos - len(positives), positives.union(negatives)
        )
        padpos.update(positives)
        padneg = cu.gen_ids_wo_overlap(
            targetneg - len(negatives), padpos.union(negatives)
        )
        padneg.update(negatives)
        assert len(padpos) == targetpos
        assert len(padneg) == targetneg
        super().__init__(padpos, padneg, fprs, k, multi_process)
