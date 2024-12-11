from threading import Thread
from queue import Queue
from record import record_audio
from stt import stt_audio
from noise_reduce import reduce_noise
from emotion3 import analyze_emotion
import sys

# record 실행
def run_record(result_queue):
    audio_file_path = record_audio()
    print(f"녹음 파일 경로 : {audio_file_path}", flush=True)
    
    print("노이즈 제거 실행", flush=True)
    reduced_audio_file_path = reduce_noise(audio_file_path)
    
    # 결과를 큐에 저장 (태그: 'record')
    result_queue.put(("record", reduced_audio_file_path))

# 감정분석 실행
def run_analyze(result_queue):
    print("감정 분석 실행", flush=True)
    emotion_result = analyze_emotion()
    
    # 결과를 큐에 저장 (태그: 'emotion')
    result_queue.put(("emotion", emotion_result))

def main():
    memberId = sys.argv[1]  # 인자로 넘어온 memberId 값 저장
    print(f"받은 멤버 ID: {memberId}", flush=True)
    print("메인 프로그램 시작", flush=True)
    
    # 쓰레드 간 결과 저장을 위한 큐 생성
    result_queue = Queue()

    # 쓰레드 생성
    record_thread = Thread(target=run_record, args=(result_queue,))
    emotion_thread = Thread(target=run_analyze, args=(result_queue,))

    # 각 쓰레드 시작
    record_thread.start()
    emotion_thread.start()

    # 쓰레드 완료 대기
    record_thread.join()
    emotion_thread.join()

    # 큐에서 결과 가져오기 (태그로 구분)
    results = {}
    while not result_queue.empty():
        tag, result = result_queue.get()
        results[tag] = result

    # 결과 할당
    reduced_audio_file_path = results.get("record")
    emotion_result = results.get("emotion")

    print("STT 실행", flush=True)
    stt_audio(reduced_audio_file_path, emotion_result, memberId)

    print("메인 프로그램 종료", flush=True)

if __name__ == "__main__":
    main()