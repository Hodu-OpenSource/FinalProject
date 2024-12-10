from record import record_audio
from stt import stt_audio

if __name__ == "__main__" : 
    print("메인 프로그램 시작")
    audio_file_path = record_audio()
    print(f"녹음 파일 경로 : {audio_file_path}")
    
    print("stt 실행")
    stt_audio(audio_file_path)
    print("메인 프로그램 종료")

