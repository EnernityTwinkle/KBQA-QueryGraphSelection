import sys
import os
from Model.Pairwise.Embedding import RelationEmbedding
from typing import List, Dict, Tuple
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)



class InputExample(object):
    """A single training/test example for simple sequence classification."""
    relationEmbedding = RelationEmbedding()
    haveRels = {}
    noRels = {}
    def __init__(self, guid, text_a, text_b=None, label=None, entitys = None, rels=[],\
                    answerType: str = '', answerStr: str = ''):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.freebaseEntity = entitys
        self.freebaseRels = rels
        self.relsId = InputExample.relation2id(self.freebaseRels)
        self.answerType = answerType
        self.answerStr = answerStr
        

    @staticmethod
    def relation2id(freebaseRels):
        relsId: List[int] = []
        for rel in freebaseRels:
            if(rel in InputExample.relationEmbedding.rel2id):
                relsId.append(InputExample.relationEmbedding.rel2id[rel])
                InputExample.haveRels[rel] = 1
            else:
                relsId.append(InputExample.relationEmbedding.rel2id['UNK'])
                # print(rel)
                InputExample.noRels[rel] = 1
        while(len(relsId) < 2):
            relsId.append(InputExample.relationEmbedding.rel2id['UNK'])
        return relsId[0:2]

