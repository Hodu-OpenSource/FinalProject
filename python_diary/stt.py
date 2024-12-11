from faster_whisper import WhisperModel
import pymysql
from datetime import datetime


#whisper 모델 설정. gpu가 아닌 cpu 레벨에서 돌아가는 단계에서는 small, medium이 적합
#cpu 환경에서는 compute_type을 int8로 하는 것을 권장함 (whisper 깃헙 참고)
model = WhisperModel("medium", device="cpu", compute_type="int8")

#DB 연결
conn = pymysql.connect(host='127.0.0.1', user='root', password='root', db='opensource', charset='utf8')
#커서 생성
cur = conn.cursor()

def stt_audio(audio_file_path):
    created_date = datetime.now()
    #해당 오디오 파일에 대해서 STT 수행. 
    #segments : 음성을 구간(segment)으로 나누고 세그먼트 별로 분석한 텍스트 결과를을 담는 객체들의 리스트
    segments, info = model.transcribe(audio_file_path,beam_size=5,language="ko")
    
    content =""
    for segment in segments:
       print(segment.text, end = "") #각 세그먼트의 변역 결과물을 출력
       content +=segment.text
    print()

    sql = """
    INSERT INTO diary (created_date, content, main_emotion, member_id)
    VALUES(%s, %s, %s, %s)
    """
    values = (created_date, content, None, None)
    
    #DB에 해석된 text 저장
    cur.execute(sql, values)
    conn.commit()
    conn.close