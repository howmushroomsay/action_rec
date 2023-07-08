# import cv2
# import requests
# import numpy as np

# url = 'http://222.20.72.172:8000/media/training/chapter/course_action/std/Arrest_Boxing/2.mp4' # Replace with your actual URL

# # Open the URL and retrieve the video data
# response = requests.get(url,stream=True)
# with open('a.mp4', 'wb') as f:
#     f.write(response.content)

# # Convert the data to a numpy array
# # nparr = np.frombuffer(data, np.uint8)

# # # Decode the array as a video stream
# # video_stream = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

# # Create a video capture object from the video stream
# video_capture = cv2.VideoCapture('a.mp4')

# if not video_capture.isOpened():
#     print("Could not open the video")
#     exit()

# while True:
#     ret, frame = video_capture.read()

#     if not ret:
#         # End of video
#         break

#     # Process and display the frame
#     cv2.imshow("Video", frame)

#     # Check for key press to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close any open windows
# video_capture.release()
# cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        # End of video
        break

    # Process and display the frame
    cv2.imshow("Video", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break