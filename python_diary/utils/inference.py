import cv2 # 얼굴 검출, 경계 상자 그리기, 텍스트 표시
import matplotlib.pyplot as plt # 	클래스별로 서로 다른 색상을 생성하기 위해 HSV 색 공간 사용
import numpy as np # 이미지 데이터 처리, 좌표 계산, 색상 데이터 변환
from keras.preprocessing import image #이미지를 로드하고 전처리하여 NumPy 배열로 변환

def load_image(image_path, grayscale=False, target_size=None):
    """
    이미지를 로드하고, 배열로 변환하는 함수.
    
    Args:
        image_path (str): 이미지 파일 경로.
        grayscale (bool): 이미지를 흑백으로 로드할지 여부 (기본값: False).
        target_size (tuple): 로드한 이미지를 리사이즈할 크기 (기본값: None).

    Returns:
        numpy.ndarray: 로드된 이미지 배열.
    """
    pil_image = image.load_img(image_path, grayscale, target_size)  # PIL 이미지를 로드
    return image.img_to_array(pil_image)  # PIL 이미지를 NumPy 배열로 변환

def load_detection_model(model_path):
    """
    얼굴 검출 모델을 로드하는 함수.
    
    Args:
        model_path (str): Haarcascade 모델 파일 경로.

    Returns:
        cv2.CascadeClassifier: OpenCV 얼굴 검출 모델 객체.
    """
    detection_model = cv2.CascadeClassifier(model_path)  # Haarcascade 모델 로드
    return detection_model

def detect_faces(detection_model, gray_image_array):
    """
    얼굴을 검출하는 함수.
    
    Args:
        detection_model (cv2.CascadeClassifier): 얼굴 검출 모델 객체.
        gray_image_array (numpy.ndarray): 입력 이미지 배열 (흑백).

    Returns:
        list: 검출된 얼굴의 좌표 리스트 [(x, y, w, h), ...].
    """
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)  # 얼굴 검출 수행

def draw_bounding_box(face_coordinates, image_array, color):
    """
    검출된 얼굴 주변에 경계 상자를 그리는 함수.
    
    Args:
        face_coordinates (tuple): 얼굴 좌표 (x, y, w, h).
        image_array (numpy.ndarray): 입력 이미지 배열.
        color (tuple): 경계 상자의 색상 (B, G, R).
    """
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)  # 경계 상자 그림

def apply_offsets(face_coordinates, offsets):
    """
    얼굴 좌표에 오프셋을 적용하여 영역 확장.
    
    Args:
        face_coordinates (tuple): 얼굴 좌표 (x, y, width, height).
        offsets (tuple): 가로 및 세로 오프셋 (x_offset, y_offset).

    Returns:
        tuple: 오프셋이 적용된 좌표 (x_min, x_max, y_min, y_max).
    """
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    """
    이미지에 텍스트를 그리는 함수.
    
    Args:
        coordinates (tuple): 텍스트를 그릴 위치 좌표 (x, y).
        image_array (numpy.ndarray): 입력 이미지 배열.
        text (str): 출력할 텍스트.
        color (tuple): 텍스트 색상 (B, G, R).
        x_offset (int): 텍스트 위치의 x축 오프셋 (기본값: 0).
        y_offset (int): 텍스트 위치의 y축 오프셋 (기본값: 0).
        font_scale (int): 텍스트 크기 (기본값: 2).
        thickness (int): 텍스트 두께 (기본값: 2).
    """
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),  # 텍스트를 이미지에 추가
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    """
    각 클래스에 대한 색상을 생성하는 함수.
    
    Args:
        num_classes (int): 클래스 수.

    Returns:
        numpy.ndarray: 생성된 색상 리스트 (B, G, R 형식).
    """
    # HSV 색 공간에서 클래스별 색상을 생성
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255  # 0-1 범위를 0-255로 변환
    return colors
