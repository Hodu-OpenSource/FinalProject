from record import record_audio
from stt import stt_audio
from noise_reduce import reduce_noise

if __name__ == "__main__" : 
    print("메인 프로그램 시작")
    audio_file_path = record_audio()
    print(f"녹음 파일 경로 : {audio_file_path}")
    
    print("노이즈 제거 실행")
    reduced_audio_file_path = reduce_noise(audio_file_path)
    
    print("stt 실행")
    stt_audio(reduced_audio_file_path)
    print("메인 프로그램 종료")


