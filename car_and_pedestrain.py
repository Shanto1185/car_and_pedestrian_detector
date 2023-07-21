import cv2


# our image
img_file = 'car2.jpg'
# video = cv2.VideoCapture('tesla dash cam.mp4')
video = cv2.VideoCapture('motor_das_cam.mp4')


# pre trained car and pedestrian classifiers
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascase_fullbody.xml'


# create car and pedetrian classifiers
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


# Run forever untill car stops
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # print(cars)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangles around the pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image with fhe faces spotted
    cv2.imshow('shanto', frame)

    # Don't autoclose (wait here in the code and listen for a key)
    cv2.waitKey(1)
