from src.cascade.filterCascade import FilterCascade
import math


class UnpaddedStatusCascade(FilterCascade):
    def __init__(
        self,
        validIds,
        revokedIds,
        p=0.5,
        k=1,
        multi_process=False,
    ):
        # Calculate FPRs
        fprs = [len(revokedIds) * math.sqrt(p) / len(validIds), p]

        # revokedIds must be positives since they are usually less
        super().__init__(revokedIds, validIds, fprs, k, multi_process)
