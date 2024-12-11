from scipy.io import loadmat # MATLAB .mat 파일을 로드하여 IMDb 데이터셋 사용
import pandas as pd # CSV 파일(fer2013.csv) 데이터를 읽고 처리
import numpy as np # 픽셀 데이터를 배열로 변환, 마스킹 및 학습-검증 데이터 분리
from random import shuffle # IMDb 데이터셋에서 데이터 샘플을 무작위로 섞음
import os # KDEF 데이터셋 디렉토리에서 이미지 파일 경로 탐색
import cv2 # 이미지 데이터를 로드하고 크기 조정, 픽셀 데이터를 처리

# 초기화
# DataManager 클래스 생성 시, 데이터셋 이름과 경로를 설정.
# 데이터 로드
# get_data 메서드를 호출하여 데이터셋 로드 및 전처리.
# FER2013의 경우 _load_fer2013에서 픽셀 데이터와 감정 레이블을 처리.
# 데이터 분리
# 전처리된 데이터를 학습 세트와 검증 세트로 분리 (split_data).
# 클래스 및 레이블 매핑
# 클래스 이름과 ID를 매핑 (get_labels 및 get_class_to_arg).

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
     # imdb: 성별 분류 데이터 셋, fer2023: 감정 분류 데이터 셋, KDEF: 감정 분류용 이미지 데이터셋
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):


    
        # 어떤 용도로 쓸거냐
        self.dataset_name = dataset_name # 데이터셋 이름
        self.dataset_path = dataset_path # 데이터셋 경로
        self.image_size = image_size # 이미지 리사이즈 할 크기
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '../datasets/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013': # 감정 분류를 사용하기 위한 데이터 셋 사용
            self.dataset_path = r"../python_diary/models/fer2013.csv"
        elif self.dataset_name == 'KDEF':
            self.dataset_path = '../datasets/KDEF/'
        else:
            raise Exception('Incorrect dataset name, please input imdb or fer2013')

    # 데이터 로딩을 위한 매서드
    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        elif self.dataset_name == 'fer2013': # 우리의 경우 이 데이터 로드만 사용
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'KDEF':
            ground_truth_data = self._load_KDEF()
        return ground_truth_data

    def _load_imdb(self): # 사용 X
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))

    # fer 2013 데이터셋 로드 및 전처리
    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path) # csv 파일 로드
        pixels = data['pixels'].tolist() # 픽셀 데이터
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels: # 픽셀 데이터를 이미지로 변환
            face = [int(pixel) for pixel in pixel_sequence.split(' ')] 
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces) # 리스트를 numpy 배열로 전환
        faces = np.expand_dims(faces, -1) # 채널 축 추가
        emotions = data['emotion'].to_numpy() # 감정 레이블 데이터를 Numpy 배열로 변환
        return faces, emotions # 전처리된 이미지 데이터와 감정 레이블 변환

    def _load_KDEF(self): # 사용 X
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions

def get_labels(dataset_name):
    if dataset_name == 'fer2013': # 7가지의 감정 라벨 가져오기
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    else:
        raise Exception('Invalid dataset name')

def get_class_to_arg(dataset_name='fer2013'): # 데이터셋 이름에 따라 클래스 이름과 인덱스 매핑을 반환
    if dataset_name == 'fer2013': # fer2013 클래스 이름과 인덱스 매핑
        return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                'surprise':5, 'neutral':6}
    elif dataset_name == 'imdb':
        return {'woman':0, 'man':1}
    elif dataset_name == 'KDEF':
        return {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}
    else:
        raise Exception('Invalid dataset name')

# 데이터셋을 학습 세트와 검증 세트로 분리
def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False): # 검증세트의 비율은 0.2
    ground_truth_keys = sorted(ground_truth_data.keys()) # 데이터셋 키를 정렬
    if do_shuffle == True: # 무작위로 섞고
        shuffle(ground_truth_keys) # 학습 데이터와 검증 데이터 분리 기준 계산
    training_split = 1 - validation_split 
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train] #학습 세트와
    validation_keys = ground_truth_keys[num_train:] # 검증세트로 키 분리
    return train_keys, validation_keys

# 일반 데이터셋을 학습 세트와 검증 세트로 분리
def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    # 학습 데이터와 검증 데이터로 분리
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    # 학습 데이터와 검증 데이터 반환
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

