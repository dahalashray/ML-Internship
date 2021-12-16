import os
import numpy as np
from facenet_image import Face

class search_face():
    def __init__(self):
        self.embedding = []
        self.embedding_name = []
        self.face_object = Face()
        
    def load_embedding(self):
        for file_ in os.listdir('faces_embedding'):
            file_path = os.path.join('faces_embedding',file_)
            x = np.load(file_path)
            self.embedding.append(x)
            self.embedding_name.append(file_)

    def compare(self, file_path):
        B = self.new_embedding(file_path)[0]
        for A,name in zip(self.embedding, self.embedding_name):
            A = A[0]
            cos_sim=np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
            print(cos_sim, name)

    def new_embedding(self,file_path):
        embedding = self.face_object.get_embedding(file_path)
        # print(embedding)
        return embedding
        


obj = search_face()
obj.load_embedding()
obj.compare('images.jpg')
# obj.compare(1)

# obj.load_embeddings()