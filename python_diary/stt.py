from faster_whisper import WhisperModel

#whisper 모델 설정. gpu가 아닌 cpu 레벨에서 돌아가는 단계에서는 small, medium이 적합
#cpu 환경에서는 compute_type을 int8로 하는 것을 권장함 (whisper 깃헙 참고)
model = WhisperModel("medium", device="cpu", compute_type="int8")

#해당 오디오 파일에 대해서 STT 수행. 해당 결과를 
segments, info = model.transcribe("C:/Users/Useong/Desktop/hodu/python_diary/audio/audio.wav",beam_size=5,language="ko")

for segment in segments:
    print(segment.text, end = "") #세그먼트 단위로 추

print(info)