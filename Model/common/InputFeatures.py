from typing import List

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, entitysId = [], relsId = [],\
                        answerTypeIds:List[int] = [], answerStrIds:List[int] = []) -> None:
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entitysId = entitysId
        self.relsId = relsId
        self.answerTypeIds = answerTypeIds
        self.answerStrIds = answerStrIds