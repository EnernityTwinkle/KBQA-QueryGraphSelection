import numpy as np



class RelationEmbedding:
    def __init__(self) -> None:
        self.relation2idFile = '/data2/yhjia/KGEmbeddings/Freebase/knowledge graphs/relation2id.txt'
        self.relation2vecFile = '/data2/yhjia/KGEmbeddings/Freebase/embeddings/dimension_50/transe/relation2vec.bin'
        self.rel2id = self.readRelation2id()
        self.embedding = self.readVec()
        self.addUNKRel()

    def readRelation2id(self):
        rel2id = {}
        try:
            with open(self.relation2idFile, 'r', encoding='utf-8') as fread:
                for line in fread:
                    lineCut = line.strip().split('\t')
                    if(len(lineCut) == 2):
                        rel2id[lineCut[0]] = int(lineCut[1])
        except:
            print('transe文件不存在')
            return rel2id
        # import pdb; pdb.set_trace()
        return rel2id


    def readVec(self):
        try:
            vec = np.memmap(self.relation2vecFile, dtype='float32', mode='r')
            embedding = np.reshape(vec, (-1, 50))
            # import pdb; pdb.set_trace()
        except:
            print('transe文件不存在')
            return np.zeros((2, 50))
        return embedding
    
    def addUNKRel(self):
        self.rel2id['UNK'] = len(self.rel2id)
        self.embedding = np.insert(self.embedding,-1,values=self.embedding[-1],axis=0)
        # import pdb; pdb.set_trace()





if __name__ == "__main__":
    relEmb = RelationEmbedding()
    sortedRel = sorted(relEmb.rel2id.items(), key = lambda x:int(x[1]))
    import pdb; pdb.set_trace()