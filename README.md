# Face Verification  
This project consists of three parts :  
1 - Face Verification :  
    This part contains code for comparing two images of faces using the InsightFace library. The goal is to determine whether the two images belong to the same person or not 
2 - Face Identification : 
First of all it's needed to create a face_bank, a .npy file containing name and the 512D feature vector of each person.  
You can add new feature vector by putting your images in face_bank folder and running the following command :  
```
python face_identification.py --update 
```  
and then run the following command to identify images :  
```
python face_identification --image YOUR_IMAGE
```  
![Sample Image](Face_Identification/output/output.png)
3 - This implementation features real-time face identification using a webcam. The system detects faces in the webcam feed and attempts to identify the person. If the face is successfully identified, the user gains access to a hand-painting program. Otherwise, an "Access Denied!" message is displayed



### Usage 
1 - git clone https://github.com/Parisa-Bagherzadeh/Face_Verification.git  
2 - Run the following command :  
```
python face_verification.py --image1 images/image1.jpg --image2 images/image2.jpg
```