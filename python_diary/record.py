import os
import pyaudio
import wave

CHUNK = 1024 #오디오 데이터를 읽는 최소 단위
FORMAT = pyaudio.paInt16 #오디오 샘플의 데이터 형식
CHANNELS = 2 #오디오 채널 수. 스테레오 녹음이기에 2로 설정
RATE = 22050 #샘플링 속도. 1초당 녹음하는 샘플의 수를 의미한다

# 저장 디렉토리 설정
print("현재 작업 디렉토리:", os.getcwd())
output_dir = "./python_diary/audio/" 
os.makedirs(output_dir, exist_ok=True)  # 디렉토리 없으면 생성

# 파일 경로 설정. output_dir로
output_file = os.path.join(output_dir, "output.wav")

p = pyaudio.PyAudio()# pyAudio 객체 생성

#오디오 스트림을 연다
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print('start recording')

frames = [] #녹음 데이터를 저장할 리스트 
seconds = 3 #녹음할 시간(초)
for i in range(0, int(RATE / CHUNK * seconds)): # 녹음 동안 읽어야할 프레임 수
    data = stream.read(CHUNK) #오디오 데이터를 읽어오고 
    frames.append(data) #저장

print('record stopped')

stream.stop_stream() #스트림을 멈추기
stream.close() #스트림 닫기
p.terminate() #pyAudio 객체 종료

# WAV 파일 생성 및 저장
wf = wave.open(output_file, 'wb')  # 같은 디렉토리 내 audio 폴더에 오디오 저장
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()