import socket
import time
from imutils.video import VideoStream
import imagezmq
import os
import cv2
import zmq
import socket
from Speaker import Speaker


speaker = Speaker(rate=70)

print("Sending request to server")

sender = imagezmq.ImageSender(connect_to="tcp://localhost:5555")

# time.sleep(50)

rpi_name = socket.gethostname()

directory = os.path.join(os.getcwd(), "rasp_img")
# while True:
i = 1
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(directory, filename))
        # sender.send_image(rpi_name, image)
        print("Sending image " + str(i) + " to server and waiting for caption")
        i += 1

        hub_reply = sender.send_image_reqrep(rpi_name, image)

        # time.sleep(300)

        hub_reply = hub_reply.decode("utf-8")
        print("Caption from server: ", hub_reply)

        speaker.textToSpeech(hub_reply)
