import os
import pyaudio
import wave
import keyboard
from datetime import datetime

def record_audio() :
    CHUNK = 1024 #오디오 데이터를 읽는 최소 단위
    FORMAT = pyaudio.paInt16 #오디오 샘플의 데이터 형식
    CHANNELS = 2 #오디오 채널 수. 스테레오 녹음이기에 2로 설정
    RATE = 22050 #샘플링 속도. 1초당 녹음하는 샘플의 수를 의미한다

    # 저장 디렉토리 설정
    save_dir = "./python_diary/audio/" 
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 없으면 생성
    file_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # 현재 시간을 오디오 파일명으로 설정
    # 파일 경로 설정. output_dir로
    # 파일 이름은 current_time으로 설정
    file_path = os.path.join(save_dir, f"{file_name}.wav")

    p = pyaudio.PyAudio()# pyAudio 객체 생성

    #오디오 스트림을 연다
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print('녹음 시작. 엔터를 눌러 녹음을 종료하세요', flush=True)

    frames = [] #녹음 데이터를 저장할 리스트 
    while not keyboard.is_pressed('q') : #엔터를 누르기 전까지 녹음 수행
        data = stream.read(CHUNK) #오디오 데이터를 읽어오고 
        frames.append(data) #저장

    print('녹음 종료', flush=True)

    stream.stop_stream() #스트림을 멈추기
    stream.close() #스트림 닫기
    p.terminate() #pyAudio 객체 종료

    # WAV 파일 생성 및 저장
    wf = wave.open(file_path, 'wb')  # 같은 디렉토리 내 audio 폴더에 오디오 저장
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"저장 경로 : {file_path}", flush=True)
    return  file_path


