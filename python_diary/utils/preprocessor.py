from imageio import imread  # 이미지 파일을 읽어오기 위한 모듈
from PIL import Image       # 이미지 처리 및 변환을 위한 라이브러리
import numpy as np          # 배열 및 수치 계산 라이브러리

def preprocess_input(x, v2=True):
    """
    이미지 데이터를 전처리하는 함수.
    
    Args:
        x (numpy.ndarray): 이미지 배열.
        v2 (bool): True이면 -0.5 ~ 0.5 범위로 스케일링, False이면 0~1 범위로 스케일링.

    Returns:
        numpy.ndarray: 전처리된 이미지 배열.
    """
    x = x.astype('float32')  # 배열을 float32로 변환
    x = x / 255.0            # 픽셀 값을 0~1로 스케일링
    if v2:
        x = x - 0.5          # 0~1 값을 -0.5 ~ 0.5로 이동
        x = x * 2.0          # 범위를 -1 ~ 1로 스케일링
    return x

def _imread(image_name):
    """
    이미지를 파일에서 읽어오는 함수.
    
    Args:
        image_name (str): 이미지 파일 경로.

    Returns:
        numpy.ndarray: 읽어온 이미지 배열.
    """
    return imread(image_name)  # 이미지를 NumPy 배열로 읽어옴

def _imresize(image_array, size):
    """
    이미지를 특정 크기로 리사이즈하는 함수.
    
    Args:
        image_array (numpy.ndarray): 입력 이미지 배열.
        size (tuple): (width, height) 형태의 리사이즈할 크기.

    Returns:
        numpy.ndarray: 리사이즈된 이미지 배열.
    """
    return np.array(Image.fromarray(image_array).resize(size))  # Pillow를 사용하여 크기 조정

def to_categorical(integer_classes, num_classes=2):
    """
    정수형 클래스를 원-핫 인코딩으로 변환하는 함수.
    
    Args:
        integer_classes (array-like): 정수형 클래스 배열.
        num_classes (int): 클래스의 총 개수 (기본값: 2).

    Returns:
        numpy.ndarray: 원-핫 인코딩된 배열 (shape: [num_samples, num_classes]).
    """
    integer_classes = np.asarray(integer_classes, dtype='int')  # 입력을 정수형 배열로 변환
    num_samples = integer_classes.shape[0]                      # 샘플 수
    categorical = np.zeros((num_samples, num_classes))          # 원-핫 배열 초기화
    categorical[np.arange(num_samples), integer_classes] = 1    # 해당 클래스 위치에 1 할당
    return categorical
