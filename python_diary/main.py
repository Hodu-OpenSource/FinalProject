from threading import Thread # 멀티 쓰레드 사용하기 위해 이거 사용
from record import record_audio
from stt import stt_audio
from noise_reduce import reduce_noise
from emotion3 import analyze_emotion 
import sys

# stt와 감정분석 멀티 쓰레딩

# stt실행
def run_stt():
    audio_file_path = record_audio()
    print(f"녹음 파일 경로 : {audio_file_path}" , flush=True)
    
    print("노이즈 제거 실행", flush=True)
    reduced_audio_file_path = reduce_noise(audio_file_path)
    
    print("stt 실행", flush=True)
    stt_audio(reduced_audio_file_path)


# 감정분석 실행

def run_analyze():
    print("감정 분석 실행", flush=True)
    emotion_result=analyze_emotion()  
    print(emotion_result)





def main() : 
    #memberId = sys.argv[1] #인자로 넘어온 memberId 값 저장
    #print(f"받은 멤버 ID: {memberId}", flush=True)
    
    print("메인 프로그램 시작", flush=True)
    
    # 쓰레드 생성
    stt_thread=Thread(target=run_stt)# 단일 인자를 받을 때 튜플로 작성하기 위한 ,
    emotion_thread=Thread(target=run_analyze)


    # 각 쓰레드 시작
    stt_thread.start()
    emotion_thread.start()

    # 쓰레드 완료 대기
    stt_thread.join()
    emotion_thread.join()
    


    print("메인 프로그램 종료", flush=True)
    
if __name__ == "__main__" :
    main()