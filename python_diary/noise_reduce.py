from scipy.io import wavfile
import noisereduce as nr
import os

def reduce_noise(audio_file_path) :
    # 저장 디렉토리 설정
    save_dir = "./python_diary/reduce_noise_audio/"
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 없으면 생성

    #노이즈 감소 녹음본 이름과 경로 설정
    file_name = os.path.basename(audio_file_path).replace(".wav", "_reduced_noise.wav")
    file_path = os.path.join(save_dir, file_name)
    
    #wav 음성 파일 읽어오기
    rate, data = wavfile.read(audio_file_path)

    # 스테레오 데이터 처리 (모노로 변환)
    #noisereduce의 노이즈 감소는 모노 데이터를 입력데이터로 받기에 변환
    if data.ndim > 1:
        data = data[:, 0]  # L 채널만 사용 (스테레오 -> 모노)

    #노이즈 감소 수행.
    #prop_decrease옵션은 노이즈 제거 강도로 기본값은 1.0
    #값이 작을수록 제거 정도가 작아지며, 음성 품질을 유지하면서 제거하는 값으로 0.8을 많이 사용
    reduce_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease = 0.8)
    
    #노이즈 감소시킨 녹음본 저장 
    wavfile.write(file_path, rate, reduce_noise)
    print("노이즈 제거 파일 저장 완료", flush=True)

    return file_path