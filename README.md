# Instagram-filters
Machine Learning

https://user-images.githubusercontent.com/91654378/152674272-502dde75-803f-4509-8c27-e1c3f6709f6c.mp4

# Description

---

Ever wondered how facial filters on social media platforms like Instagram really work? What tech did Instagram use to create real-time face filters?

You’ve probably seen face filters invading Instagram over the past few years. Just open the camera, point it towards your face, and the app will create a random filter for you.

Instagram face filters are the most common across social media, although they take different forms. Instagram offers beautification filters along with its augmented-reality facial filters, like those that add a cat’s ears and tongue to a person’s face. Snapchat also offers a gallery of filters where users can swipe through beauty effects on their selfie camera. Beauty filters in Tik-Tok’ and snapchat’s app are part of an app setting called “Enhance,” where users can set a standard beautification on any subject.

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Instagram-filters/blob/main/images/img1.png" />

# Facial Filters

---

Face filters are augmented reality effects that overlay virtual items on the facial and are enabled by face identification technology. Face filters were first introduced by Snapchat in 2015 as a fun way to dress up your image. Since then, they've evolved into a useful tool for improving digital communication and connection.

You can superimpose 3D items on the face or backgrounds, animate facial expressions, test on virtual merchandise, and apply AR makeup, among other things. Users may convert themselves into anyone or anything using augmented reality, turning selfies into a storytelling journey.

# Face Detection

---

Face detection is a term that refers to computer technology that can detect the presence of people's faces in digital photographs. Face detection programmed function by finding human faces within larger photos using machine learning and calculations known as algorithms. A facial detection system uses biometrics to map facial features from a photograph or video.

Face detection algorithms frequently start by looking for human eyes, which is a complicated procedure. Eyes are one of the easiest
things to recognize since they form a valley region. After detecting eyes, the algorithm may try to detect other facial features such as
brows, mouth, nose, nostrils, and iris. Once the algorithm has determined that a facial region has been recognised, it can do additional tests to confirm that it has indeed detected a face.
<p align="center">
<img alt="Visual Studio Code" height="500px" width="500px" src="https://github.com/harshithvh/Instagram-filters/blob/main/images/img2.png" >
</p><br>

# Face Recognition

---

Facial recognition is a method of recognizing or verifying a person's identification by looking at their face. People can be identified in pictures, films, or in real time using facial recognition technology. Biometric security includes facial recognition.

Face recognition software use computer algorithms to identify specific, distinguishing features on a person's face. These features, such as eye distance or chin shape, are then transformed into a mathematical representation and compared to data from other faces in a face recognition database.

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Instagram-filters/blob/main/images/img3.png" />

It simply means that the face detection system can detect the presence of a human face in a video clip but cannot identify the individual. Face detection is a part of facial recognition systems; the first step in facial recognition is identifying the existence of a human face.

# Facial Landmarks Detection

---

Facial landmarks are used to align facial images to a mean face shape, so that after alignment the location of facial landmarks in all images is approximately the same. However it makes sense that facial recognition algorithms trained with aligned images would perform much better, and this intuition has been confirmed by many research papers.
<p align="center">
<img alt="Visual Studio Code" height="650px" width="650px" src="https://github.com/harshithvh/Instagram-filters/blob/main/landmarks.jpg" >
</p><br>
  
# Feature Extraction

---

In the real world, all the data we collect are in large amounts. To understand this data, we need a process. Manually, it is not possible to process them. This is where the concept of feature extraction in machine learning comes into picture.

Feature extraction in image processing is a part of the dimensionality reduction process, in which an initial set of the raw data is divided and reduced to more manageable groups. So when you want to process that data, it will be easier. The most promising characteristic of these large data sets is that they have a large number of variables. These variables require a lot of computing resources to process them. So Feature extraction helps to get the best feature from those big data sets by selecting and combining variables into features, thus, effectively reducing the amount of data. 

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Instagram-filters/blob/main/images/img6.png" />

# Approaches to perform Face Detection
# Traditional Machine Learning

---

These algorithms learn from data, with subject matter experts selecting the algorithm and features (inputs) to be fed into the

algorithm. Traditional machine learning models require all inputs to be in the form of structured data, such as numbers. Classification, regression, clustering, and dimensionality reduction challenges can all be solved with traditional machine learning methods.

Traditional Machine Learning methods are very fast, accurate but can't work in every lighting condition and in case of face detection it is accurate only when person is facing the camera.

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Instagram-filters/blob/main/images/img4.png" />

# Deep Learning

---


These algorithms can learn important aspects from underlying data and determine which features to pay attention to without the need for experts to identify them directly.

Unstructured data, such as photographs and videos, can be fed into deep learning models.

Suppose if you want to teach a neural network to recognize a cat, for instance, you don’t tell it to look for whiskers, ears, fur, and eyes.

You simply show it thousands and thousands of photos of cats, and eventually it works things out.

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Instagram-filters/blob/main/images/img5.png" />

Deep Learning methods are slow, very accurate, can work in every lighting condition and in case of face detection Can work with any face rotation.

In deep learning approach we require huge amount of data to train our model for good accuracy, it also requires large computation power to process the information storing data which also take more time due to these reason traditional machine learning approach preferred over deep learning approach.
