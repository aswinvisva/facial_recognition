# Facial Recognition using Machine Learning

This project uses the TensorFlow InceptionV3 architecture with ImageNet weights to predict a feature vector representing the faces of different people. A region of interest is taken by using OpenCV Haar Cascade and then the feature vectors are compared to the existing faces using cosine similarity, providing a probability distribution of whose face is in the image.

Ex.

When the following image of Shaquille O'Neal was passed to the system:
![alt-text](https://github.com/aswinvisva/facial_recognition/blob/master/images/shaq_test.jpg)

The returned probability distribution was:
{'Shaq': 0.7704336, 'Emma Watson': 0.47395822, 'Jaime Foxx': 0.62511873, 'Lionel Messi': 0.5163602}
