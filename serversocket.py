import os
import socket
import cv2
import numpy
import base64
import glob
import sys
import time
import threading
from datetime import datetime

from tensorflow.keras.utils import img_to_array
import imutils
from keras.models import load_model
from gtts import gTTS
from playsound import playsound

global voice_flag
voice_flag=0
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 데이터 및 이미지 로드를 위한 매개 변수
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# 경계 상자 모양에 대한 하이퍼 매개 변수
# 모델 불러오기
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))
def play_voice(text):
        global voice_flag
        if voice_flag == 0 :
            voice_flag = 1
            playsound(text+'.mp3')
            time.sleep(5)
            voice_flag = 0    

    
class ServerSocket:

    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.createImageDir()
        self.folder_num = 0
        self.socketOpen()
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.start()

    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))
        self.sock.listen(1)
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is open')
        self.conn, self.addr = self.sock.accept()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is connected with client')

    def receiveImages(self):
        cnt_str = ''
        cnt = 0

        try:
            while True:
                if (cnt < 10):
                    cnt_str = '000' + str(cnt)
                elif (cnt < 100):
                    cnt_str = '00' + str(cnt)
                elif (cnt < 1000):
                    cnt_str = '0' + str(cnt)
                else:
                    cnt_str = str(cnt)
                if cnt == 0: startTime = time.localtime()
                cnt += 1
                
                length = self.recvall(self.conn, 64)
                length1 = length.decode('utf-8')
                stringData = self.recvall(self.conn, int(length1))
                stime = self.recvall(self.conn, 64)
                print('send time: ' + stime.decode('utf-8'))
                now = time.localtime()
                print('receive time: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1)


                # data를 디코딩한다.
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                
                frame = imutils.resize(frame,width=300)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
                canvas = numpy.zeros((250, 300, 3), dtype="uint8")
                frameClone = frame.copy()
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True,
                    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = faces
         # 영상에서 얼굴 추출, 크기를 고정된 28x28픽셀로 조정한 다음 준비합니다.
            # CNN을 통한 분류에 대한 ROI
                    roi = gray[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = numpy.expand_dims(roi, axis=0)
        
        
                    preds = emotion_classifier.predict(roi)[0]
                    emotion_probability = numpy.max(preds)
                    label = EMOTIONS[preds.argmax()]
                    answer_str=label
                    t = threading.Thread(target=play_voice, args=(answer_str,))
                    t.start()
                else: continue

 
                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]

                
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


                cv2.imshow('your_face', frameClone)
                
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if (cnt == 60 * 10):
                    cnt = 0
                    convertThread = threading.Thread(target=self.convertImage(str(self.folder_num), 600, startTime))
                    convertThread.start()
                    self.folder_num = (self.folder_num + 1) % 2
        except Exception as e:
            print(e)
            self.convertImage(str(self.folder_num), cnt, startTime)
            self.socketClose()
            cv2.destroyAllWindows()
            self.socketOpen()
            self.receiveThread = threading.Thread(target=self.receiveImages)
            self.receiveThread.start()

    def createImageDir(self):

        folder_name = str(self.TCP_PORT) + "_images0"
        try:
            if not os.path.exists(folder_name):
                os.makedirs(os.path.join(folder_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create " + folder_name +  " directory")
                raise

        folder_name = str(self.TCP_PORT) + "_images1"
        try:
            if not os.path.exists(folder_name):
                os.makedirs(os.path.join(folder_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create " + folder_name + " directory")
                raise

        folder_name = "videos"
        try:
            if not os.path.exists(folder_name):
                os.makedirs(os.path.join(folder_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create " + folder_name + " directory")
                raise
         


    def convertImage(self, fnum, count, now):
        global size
        img_array = []
        cnt = 0
        for filename in glob.glob('./' + str(self.TCP_PORT) + '_images' + fnum + '/*.jpg'):
            if (cnt == count):
                break
            cnt = cnt + 1
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        
        file_date = self.getDate(now)
        file_time = self.getTime(now)
        name = 'video(' + file_date + ' ' + file_time + ').mp4'
        file_path = './videos/' + name
        

        for i in range(len(img_array)):
            out.write(img_array[i])
            out.release()
            print(u'complete')

    def getDate(self, now):
        year = str(now.tm_year)
        month = str(now.tm_mon)
        day = str(now.tm_mday)

        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        return (year + '-' + month + '-' + day)

    def getTime(self, now):
        file_time = (str(now.tm_hour) + '_' + str(now.tm_min) + '_' + str(now.tm_sec))
        return file_time
    
    
    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf    
    

def main():
    server = ServerSocket('192.168.0.18', 8080)

if __name__ == "__main__":
    main()
    
 