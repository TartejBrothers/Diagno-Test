import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import threading


def input_thread():
    global user_input
    user_input = input("Enter the letters:")


user_input = ""

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
fontSize = 2


def display_text(text, face, img):
    global fontSize
    cvzone.putTextRect(
        img, "READ THE FOLLOWING", (face[10][0] - 200, face[10][1] - 100), scale=2
    )
    cvzone.putTextRect(img, text, (face[10][0] - 100, face[10][1] - 50), scale=fontSize)
    cv2.imshow("Image", img)
    # cv2.waitKey(2000)  # Display the text for 2 seconds


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 725
        d = (W * f) / w
        fontSize = 2

        if d < 36:
            pass
        else:
            for _ in range(4):
                text = "AHDQLCO"
                display_text(text, face, img)

                # Create and start a separate thread for input
                input_thread_obj = threading.Thread(target=input_thread)
                input_thread_obj.start()

                # Wait for the user input thread to finish
                input_thread_obj.join()

                if user_input == text:
                    fontSize -= 0.5
                    print(f"Correct! Font size reduced to {fontSize}")
                else:
                    print("Incorrect! Test stopped.")
                    break

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
