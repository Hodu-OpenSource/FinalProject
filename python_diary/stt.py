from faster_whisper import WhisperModel
from datetime import datetime
import os

#whisper 모델 설정. gpu가 아닌 cpu 레벨에서 돌아가는 단계에서는 small, medium이 적합
#cpu 환경에서는 compute_type을 int8로 하는 것을 권장함 (whisper 깃헙 참고)
model = WhisperModel("medium", device="cpu", compute_type="int8")

def stt_audio(audio_file_path, audio_file_name) :
    save_dir = "./python_diary/text/" 
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 없으면 생성

    text_file_path = os.path.join(save_dir, f"{audio_file_name}.txt")

    #해당 오디오 파일에 대해서 STT 수행. 
    #segments : 음성을 구간(segment)으로 나누고 세그먼트 별로 분석한 텍스트 결과를을 담는 객체들의 리스트
    segments, info = model.transcribe(audio_file_path,beam_size=5,language="ko")
    
    with open(text_file_path, "w", encoding="utf-8") as file:
        for segment in segments:
            print(segment.text, end = "") #각 세그먼트의 변역 결과물을 출력
            file.write(segment.text)
            
    print()