import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Detector:

    def __init__(self):
        self.roi = None
        self.face_vectors = {}
        self.method = 1
        self.model = None

    def get_roi(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if(len(faces) > 1):
            raise Exception("There are multiple faces in the image!")


        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cv2.imshow("Face", roi_gray)
            cv2.waitKey(0)

        self.roi = roi_gray

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
            clf.fit(X,Y)
            self.model = clf
            des = len(self.face_vectors)

        print(des)
        return des

    def add_face(self, image, name):
        self.get_roi(image)
        face = self.get_keypoints()

        self.face_vectors[name] = face

    def check_face(self, image):
        if(len(self.face_vectors) == 0): return False

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

        for key, face in self.face_vectors.items():
            dist = np.linalg.norm(face - des)
            print(dist)


if __name__ == '__main__':
    shaq_image = cv2.imread("images/shaq.jpg")
    emma_watson_image = cv2.imread("images/shaq_test.jpg")

    dct = Detector()
    dct.add_face(shaq_image, "Shaq")
    dct.check_face(emma_watson_image)