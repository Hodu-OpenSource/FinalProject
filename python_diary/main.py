from record import record_audio
from stt import stt_audio
from noise_reduce import reduce_noise
import sys

def main() : 
    memberId = sys.argv[1] #인자로 넘어온 memberId 값 저장
    print(f"받은 멤버 ID: {memberId}", flush=True)
    
    print("메인 프로그램 시작", flush=True)
    audio_file_path = record_audio()
    print(f"녹음 파일 경로 : {audio_file_path}" , flush=True)
    
    print("노이즈 제거 실행", flush=True)
    reduced_audio_file_path = reduce_noise(audio_file_path)
    
    print("stt 실행", flush=True)
    stt_audio(reduced_audio_file_path, memberId)
    print("메인 프로그램 종료", flush=True)
    
if __name__ == "__main__" :
    main()