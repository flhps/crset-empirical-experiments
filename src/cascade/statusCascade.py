from src.cascade.paddedCascade import PaddedCascade
import math


class StatusCascade(PaddedCascade):
    def __init__(
        self,
        validIds,
        revokedIds,
        rHat,
        p=0.5,
        k=1,
        multi_process=False,
    ):
        fprs = [math.sqrt(p) / 2.0, p]
        super().__init__(validIds, revokedIds, rHat, 2 * rHat, fprs, k, multi_process)
