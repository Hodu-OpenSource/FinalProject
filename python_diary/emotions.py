import cv2  # 이미지 처리 및 변환과 얼굴 검출, 시각화에 사용
import numpy as np # 이미지 데이터를 사용하여 배열 계산과 데이터 전처리에 사용
from keras.models import load_model # 모델 로드하는데 사용
from statistics import mode # 최빈값 출력해내기 위해 파이썬 자체의 라이브러리 사용
from utils.datasets import get_labels # 학습 데이터셋 로드 및 전처리
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets # 얼굴 검출 및 시각화
from utils.preprocessor import preprocess_input # 이미지를 모델 입력 형식으로 변환



USE_WEBCAM = True  # True면 웹캠을 쓰고, False면 동영상 파일을 사용

# 데이터 가져오기 위한 경로 설정
# emotion_model.hdf5 모델은 감정을 분류하는 딥러닝 모델 
# angny, disgust, fear, happy, sad, surprise, neutral 제공
emotion_model_path = r"C:\Users\i\Desktop\emotion\Emotion\models\emotion_model.hdf5" # 현재 컴파일 환경 기준

# 사용한 데이터 셋 -> 48*48 크기의 흑백 얼굴 이미지를 포함하며, 감정 분류를 위한 공개 데이터 셋
# 35,887개의 이미지가 있고, 7개의 감정 레이블로 구성되어있음
emotion_labels = get_labels('fer2013')

# 얼굴 영역 잡기 위한 설정
frame_window = 10 # 최근 10개의 감정 결과를 기반으로 평균 계산 하고
emotion_offsets = (20, 40) # 감지된 얼굴에 영역에 대해 여유공간 추가

# 얼굴 검출 및 감정 모델 가져오기
# 오픈CV에서 제공하는 얼굴 검출 코드 반환
# Haar-like Feature를 사용하여 물체를 감지하는 OpenCV의 객체 검출 기술
# haarcascade_frontalface_default.xml은 정면 얼굴 검출을 위한 Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(r"C:\Users\i\Desktop\emotion\Emotion\models\haarcascade_frontalface_default.xml") # 현재 컴파일 환경 기준
# 모델 로드
emotion_classifier = load_model(emotion_model_path, compile=False)

# 모델 재컴파일 -> 데이터와 모델이 조금 옛날 버전이라 현재의 이전 버전의 Keras/TensorFlow에서 학습되었기에 로드를 다시 해줌
emotion_classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델에서 가져온 입력의 크기를 확인하고
emotion_target_size = emotion_classifier.input_shape[1:3]

# 감정 결과 저장할 리스트 초기화
emotion_window = []

# 비디오 시작
cv2.namedWindow('window_frame') #Open CV 창을 생성
cap = cv2.VideoCapture(0) if USE_WEBCAM else cv2.VideoCapture('./demo/dinner.mp4')

while cap.isOpened(): # 캠이 열려있는 동안 반복하고
    ret, bgr_image = cap.read() # 프레임을 읽어서(ret)
    if not ret: # 프레임 읽기가 실패(not ret)하면 루프 종료하고
        break

    # 그레이스케일은 단순히 흑백의 밝기만을 사용하는 방식
    # 읽은 프레임을 그레스케일과 RGB로 변환
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) # 얼굴 검출에 사용될 이미지 생성
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) # RGB 형식으로 변환하여 감정 분석 및 화면 출력에 사용

    # 얼굴 검출 수행하고
    faces = face_cascade.detectMultiScale(
        gray_image, # 얼굴 검출에 사용할 이미지
        scaleFactor=1.1, # 얼굴 크기 축소 비율
        minNeighbors=5, # 최소 검출 영역 크기
        minSize=(30, 30), # 검출할 얼굴의 최소 크기
        flags=cv2.CASCADE_SCALE_IMAGE, # 고정된 크기에서 스케일링된 이미지 검출
    )

    # 검출된 얼굴에 대해 감정 분류 수행
    for face_coordinates in faces: # 검출된 얼굴들에 대해 반복
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets) # 얼굴 영역에 여유 공간을 추가하여 조금 더 넓은 영역을 선택
        gray_face = gray_image[y1:y2, x1:x2] # 얼굴 영역만 잘라내기

        try:
            gray_face = cv2.resize(gray_face, emotion_target_size) # 얼굴 이미지는 모델과 같은 사이즈로 리사이즈
        except Exception as e: # 안되면
            continue # 패스하고

        # 입력받은 데이터를 전처리
        gray_face = preprocess_input(gray_face, True) 
        # 모델에 입력하기 위해 차원 추가
        gray_face = np.expand_dims(gray_face, 0) 
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face) # 감정 분류 모델을 사용해 예측 수행
        emotion_probability = np.max(emotion_prediction) # 예측된 확률 중 가장 높은 값을 추출
        emotion_label_arg = np.argmax(emotion_prediction) # 가장 높은 확률의 인덱스를 추출 -> 이 인덱스 안에는 특정 감정이 있음
        emotion_text = emotion_labels[emotion_label_arg] # 인덱스를 감정 레이블로 변환
        emotion_window.append(emotion_text) # 결과를 감정 리스트에 넣어줌

        # 최근 결과 리스트 크기가 설정된 프레임 수를 초과하면 
        if len(emotion_window) > frame_window:
            emotion_window.pop(0) #  오래된 결과 삭제
        try:
            emotion_mode = mode(emotion_window) # 결과에서 가장 빈번한 감정을 계산
        except Exception as e: # 그게 아니면
            continue # 넘어감

        # 감정마다 색깔 주기
        color_map = {
            'angry': (255, 0, 0),
            'sad': (0, 0, 255),
            'happy': (255, 255, 0),
            'surprise': (0, 255, 255),
        }
        # 감정에 따라 색깔 변경
        color = np.asarray(color_map.get(emotion_text, (0, 255, 0))) * emotion_probability
        color = color.astype(int).tolist()
        # 얼굴 경계 상자 그리기
        # 얼굴 좌표(face_coordinates)를 사용하여 RGB 이미지 위에 감정에 따른 색상의 경계 상자를 그림
        draw_bounding_box(face_coordinates, rgb_image, color)

        # 감정 텍스트(emotion_mode) 표시 색상 넣어 표시
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
    # open cv 사용하기위해 RGB에서 BGR로 변환
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image) # 결과를 open cv 창에 표시

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 끝
cap.release()
cv2.destroyAllWindows()
