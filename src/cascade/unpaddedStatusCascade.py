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
        fprs = [math.sqrt(p) / 2.0, p]
        
        # ValidIds are positives, revokedIds are negatives
        if len(validIds) >= len(revokedIds):
            raise ValueError("Number of valid IDs must be less than number of revoked IDs")
        super().__init__(validIds, revokedIds, fprs, k, multi_process)