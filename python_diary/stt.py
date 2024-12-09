from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8")

segments, info = model.transcribe("C:/Users/Useong/Desktop/hodu/python_diary/audio2.wav",beam_size=5,language="ko")

for segment in segments:
    print(segment.text, end = "")
