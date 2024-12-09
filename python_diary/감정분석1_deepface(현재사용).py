import cv2  # OpenCV 라이브러리로, 컴퓨터 비전 작업을 위한 이미지 및 비디오 처리 도구 제공
import mediapipe as mp  # MediaPipe 라이브러리로, 얼굴 검출, 동작 추적 등을 지원하는 ML 기반 유틸리티
from deepface import DeepFace  # DeepFace 라이브러리로, 얼굴 인식 및 감정 분석 기능 제공




# deepface는 MIT 기반 오픈소스, 나이, 성별 예측 등 도 있지만
# 당장 여기선 감정분석을 위해 사용, 이미 학습된 딥러닝 모델을 활용

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection  # 얼굴 검출을 위한 MediaPipe 솔루션 객체
mp_drawing = mp.solutions.drawing_utils  # 검출 결과를 시각적으로 표시하는 도구

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 기본 웹캠(카메라 ID 0)에서 비디오 스트림을 가져옴

# 해상도 설정 (1280x720으로 설정) --> 이거 오류 뜰떄가 많아서 일단 보류처리
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 카메라 영상의 가로 해상도를 1280 픽셀로 설정
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 카메라 영상의 세로 해상도를 720 픽셀로 설정

print("웹캠을 시작합니다. ESC를 눌러 종료하세요.")

# MediaPipe 얼굴 검출과 DeepFace 감정 분석을 사용
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():  # 웹캠이 열려 있는 동안 계속 실행
        ret, frame = cap.read()  # 웹캠에서 프레임 읽기
        if not ret:  # 프레임 읽기 실패 시
            print("카메라를 읽을 수 없습니다!")  # 오류 메시지 출력
            break

        # 프레임을 BGR에서 RGB로 변환 (MediaPipe는 RGB 이미지를 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV 기본은 BGR 형식이므로 변환 필요
        results = face_detection.process(rgb_frame)  # MediaPipe로 얼굴 검출 수행

        # 얼굴이 검출된 경우 처리
        if results.detections:
            for detection in results.detections:
                # 얼굴 검출 영역 좌표 가져오기 (상대적 좌표로 반환됨)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape  # 현재 프레임의 높이, 너비 가져오기
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # 얼굴 영역 추출 (프레임에서 검출된 얼굴 부분만 잘라내기) 네모 모양으로
                face_img = frame[y:y + h, x:x + w]

                # DeepFace로 감정 분석
                try:
                    # DeepFace로 감정 분석 수행 (주요 감정: happy, sad, angry 등)
                    analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False) # 여기서 deepface가 분석한 감정을 받아서
                    if isinstance(analysis, list):  # 분석 결과가 리스트 형식일 경우 처리
                        analysis = analysis[0]  # 첫 번째 요소 가져오기
                    emotion = analysis.get('dominant_emotion', 'Unknown')  # 예측된 감정은 dominat_emotion 필드로 제공
                except Exception as e:
                    # 예외 발생 시 감정을 Unknown으로 설정
                    emotion = "Unknown"

                # 얼굴 영역에 사각형 표시 및 감정 텍스트 출력
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 검출된 얼굴에 파란색 사각형 그리기
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # deepface로 받아온 감정 여기다 출력

        # 화면에 결과 출력
        cv2.imshow('Emotion Detection', frame)  # 'Emotion Detection' 창에 실시간 결과 표시

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키(코드 27) 입력 확인
            break

# 프로그램 종료 및 자원 해제
cap.release()  # 웹캠 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기


# deepface에서 제공해주는 감정들은 happy(행복), sad(슬픔), angry(분노), fear(두려움), disgust(혐오), surprise(놀람), neutral(중립)
# 써봤는데 거의 netural만 나오고 happy정돈 잘 인식하는데 sad 부분은 인식을 잘 못함