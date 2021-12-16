import os  
import cv2
import uuid
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model


class Face:
  def __init__(self):
    self.model = load_model('model/facenet_keras.h5')
    self.detector = MTCNN()
    self.model.load_weights('model/facenet_keras_weights.h5')

  def from_folder(self, folder):
    for subdir in os.listdir(folder):
      for files in os.listdir(os.path.join(folder,subdir)):
        file_path = os.path.join(folder,subdir,files)
        # print(file_path)
        face = self.get_face(file_path)
        embedding = self.get_embedding(file_path)
        # print('Face:{}\n Embedding:{}'.format(face, embedding))
      
  def get_face(self,file_path, required_size=(160,160)):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.asarray(image)
    # print('pixel:{}'.format(pixels))
    results = self.detector.detect_faces(pixels)
    # print('results:{}'.format(results))
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    face_array = np.asarray(face)
    return face_array

  def get_embedding(self, file_path):
    face_array = self.get_face(file_path)
    image = face_array.astype('float32')
    image = image/255.0
    train_image = np.expand_dims(image, axis=0)
    y_pred = self.model.predict(train_image)
    # print('embedding shape:{}'.format(y_pred.shape))
    return y_pred

  def save_(self, face_array, face_embedding):
    os.makedirs('faces_cropped', exist_ok=True)
    os.makedirs('faces_embedding', exist_ok=True)
    # face_array
    face_cropped = cv2.resize(face_array,(160,160))
    face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_RGB2BGR)
    uid = str(uuid.uuid4())
    image_name = os.path.join('faces_cropped', uid + '.png')
    embedding_name = os.path.join('faces_embedding', uid + '.npy')
    cv2.imwrite(image_name, face_cropped)
    np.save(embedding_name, face_embedding)
  
  def save_faces_embedding(self, folder):
    for subdir in os.listdir(folder):
      for files in os.listdir(os.path.join(folder,subdir)):
        file_path = os.path.join(folder,subdir,files)
        # print(file_path)
        faces = self.get_face(file_path)
        embedding = self.get_embedding(file_path)
        self.save_(faces, embedding)
    
if __name__ == '__main__':
  obj = Face()
  # print(obj.model.summary())
  # print(obj.detector)
  obj.save_faces_embedding('data_dota')
  # s = obj.get_embedding(file_path_1)

  # A = obj.get_embedding(file_path_1)[0]
  # B = obj.get_embedding(file_path_2)[0]

  # cos_sim=np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
  # print(cos_sim)


  