import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2 as cv

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Detector:

    def __init__(self):
        self.roi = None
        self.face_vectors = {}
        self.method = 3
        self.model = None

    def get_roi(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if (len(faces) > 1):
            raise Exception("There are multiple faces in the image!")

        for (x, y, w, h) in faces:
            roi = image[y:y + h, x:x + w]
            cv2.imshow("Face", roi)
            cv2.waitKey(0)

        self.roi = roi

    def get_keypoints(self):

        if self.method == 0:
            sift = cv2.xfeatures2d.SIFT_create(128)
            kp, des = sift.detectAndCompute(self.roi, None)
            keypoints = cv2.drawKeypoints(self.roi, kp, self.roi)
            self.model = sift
            cv2.imshow("Keypoints", keypoints)
            cv2.waitKey(0)
        elif self.method == 1:
            pca = PCA(n_components=128)
            print(len(np.array(self.roi).flatten()))
            pca.fit(np.array(self.roi))
            self.model = pca
            des = pca.singular_values_
            print(len(des))
        elif self.method == 2:
            X = np.array(self.roi).flatten().reshape(1, -1)
            Y = [len(self.face_vectors)]
            clf = LinearDiscriminantAnalysis()
            clf.fit(X, Y)
            self.model = clf
            des = len(self.face_vectors)
        elif self.method == 3:
            xception_base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(128, 128, 3),
                                              pooling='avg')

            for layer in xception_base_model.layers:
                layer.trainable = False

            x = Flatten()(xception_base_model.output)

            self.model = Model(inputs=xception_base_model.input, outputs=x)
            self.model.summary()

            img = cv.resize(self.roi, (128, 128))
            img = img.reshape(1, 128, 128, 3)

            des = self.model.predict(preprocess_input(img))

        print(des)
        return des

    def add_face(self, image, name):
        self.get_roi(image)
        face = self.get_keypoints()

        self.face_vectors[name] = face

    def check_face(self, image):
        if len(self.face_vectors) == 0:
            return False

        self.get_roi(image)

        if self.method == 0:
            sift = cv2.xfeatures2d.SIFT_create(128)
            kp, des = sift.detectAndCompute(self.roi, None)
            keypoints = cv2.drawKeypoints(self.roi, kp, self.roi)
            cv2.imshow("Keypoints", keypoints)
            cv2.waitKey(0)
        elif self.method == 1:
            pca = PCA(n_components=128)
            pca.fit(np.array(self.roi))
            des = pca.singular_values_
        elif self.method == 2:
            self.model.predict(np.array(self.roi).reshape(-1, 1))
        elif self.method == 3:
            img = cv.resize(self.roi, (128, 128))
            img = img.reshape(1, 128, 128, 3)
            des = self.model.predict(preprocess_input(img))

        for key, face in self.face_vectors.items():
            dist = np.linalg.norm(face - des)
            print("Testing against: %s, distance: %s" % (key, str(dist)))
            print(dist)


if __name__ == '__main__':
    shaq_image = cv2.imread("images/shaq.jpg")
    shaq_test = cv2.imread("images/shaq_test.jpg")
    jaime_test = cv2.imread("images/jaime_fox_test.jpg")
    jaime_image = cv2.imread("images/jaime_fox.jpg")
    watson_test = cv2.imread("images/emma_watson_test.jpg")
    watson_image = cv2.imread("images/emma_watson.jpg")

    dct = Detector()
    dct.add_face(shaq_image, "Shaq")
    dct.add_face(watson_image, "Emma Watson")
    dct.add_face(jaime_image, "Jaime Foxx")

    dct.check_face(watson_test)
    dct.check_face(jaime_test)
    dct.check_face(shaq_test)
    dct.check_face(jaime_test)
