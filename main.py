import cv2
import dlib
import pickle
import numpy as np
import math

detector = dlib.get_frontal_face_detector() #Face detector
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks_with_point(image, frame):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #get facial landmarks with prediction model
        shape = model(image, d)
        xpoint = []
        ypoint = []
        for i in range(17, 68):
            #if (i == 27) | (i == 30):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        #center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        #Calculate distance between particular points and center point
        xdistcent = [(x-xcenter) for x in xpoint]
        ydistcent = [(y-ycenter) for y in ypoint]

        #prevent divided by 0 value
        if xpoint[11] == xpoint[14]:
            angle_nose = 0
        else:
        #point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ypoint[11]-ypoint[14])/(xpoint[11]-xpoint[14]))*180/math.pi)

        #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx,cy,x,y in zip(xdistcent, ydistcent, xpoint, ypoint):
        #Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

        #Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter,xcenter))
            centpar = np.asarray((y,x))
            dist = np.linalg.norm(centpar-meanar)

        #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y-ycenter)/(x-xcenter))*180/math.pi) - angle_nose
                #print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        #If no face is detected set the data to value "error" to catch detection errors
        landmarks = "error"
    return landmarks

def show_image_test(model, emotions, imagelocation):
    training_data = []
    imageloc = imagelocation
   
    #imagefile = request.files['image']
    #filename = werkzeug.utils.secure_filename(imagefile.filename)
    #print("\nReceived image File name : " + imagefile.filename)
    #imagefile.save("D:\COLLEGE\S8\Project\EMOTION DETECTION\saved" + filename)
        
    #print(imageloc)
    image = cv2.imread(imageloc)

    test_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(test_img)
    #plt.show()

    test_img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(test_img_gray, cmap='gray')
    #plt.show()

    haar_cascade_face = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    faces_rects = haar_cascade_face.detectMultiScale(test_img_gray, scaleFactor = 1.2, minNeighbors = 5)
    faces_rects

    #print('Faces found: ', len(faces_rects))

    if(len(faces_rects) < 1):
        #print('No face detected')
        return 'no_face'
    elif(len(faces_rects) > 1):
        #print('Too many faces detected')
        return 'many_face'
    else:
        for (x,y,w,h) in faces_rects:
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop_img = image[y:y+h, x:x+w]
            #crop1_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            #plt.imshow(crop1_img)
            #plt.show()
            #cv2_imshow(crop_img)
            #cv2.waitKey(0)

        #plt.imshow(test_img)
        #plt.show()

        #image.thumbnail((160, 160))
        crop_img = cv2.resize(crop_img, (240, 240))

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        #Get Point and Landmarks
        landmarks_vectorised = get_landmarks_with_point(clahe_image, crop_img)
        #print(landmarks_vectorised)
        if landmarks_vectorised == "error":
            pass
        else:
            #Predict emotion
            training_data.append(landmarks_vectorised)
            npar_pd = np.array(training_data)
            prediction_emo_set = model.predict_proba(npar_pd)
            if cv2.__version__ != '3.1.0':
                prediction_emo_set = prediction_emo_set[0]
                print(zip(model.classes_, prediction_emo_set))
                prediction_emo = model.predict(npar_pd)
            if cv2.__version__ != '3.1.0':
                prediction_emo = prediction_emo[0]
                #print(emotions[prediction_emo])
                return emotions[prediction_emo]

        #crop2_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        #plt.imshow(crop2_img)
        #plt.show()
        #cv2_imshow(crop_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
