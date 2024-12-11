package hodu.diary.service;

import hodu.diary.domain.Diary;
import hodu.diary.dto.DiaryDTO;
import hodu.diary.repository.DiaryRepository;
import hodu.member.domain.Member;
import hodu.member.exception.MemberNotFoundException;
import hodu.member.repository.MemberRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

@Service
public class DiaryService {
    private final DiaryRepository diaryRepository;
    private final MemberRepository memberRepository;
    private final String scriptPath = "../python_diary/main.py";

    public DiaryService(DiaryRepository diaryRepository, MemberRepository memberRepository) {
        this.diaryRepository = diaryRepository;
        this.memberRepository = memberRepository;
    }

    @Transactional
    public void addDiary(Long memberId) {
        Member member = memberRepository.findById(memberId)
                .orElseThrow(()->new MemberNotFoundException("Id : " + memberId + "인 멤버를 찾을 수 없습니다"));
        
        try{
            ProcessBuilder processBuilder = new ProcessBuilder("python", scriptPath, Long.toString(memberId));
            processBuilder.redirectErrorStream(true); // stderr를 stdout에 병합

            processBuilder.environment().put("PYTHONIOENCODING", "UTF-8"); // 한글이 정상적으로 저장되고 출력되도록 PYTHONIOENCODING 환경 변수 설정

            Process process = processBuilder.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String line;
            while((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("파이썬 스크립트 정상적으로 종료");
            } else {
                System.err.println("파이썬 스크립트 비정상적인 종료. exitCode : " + exitCode);
            }

        } catch (IOException e) {
            System.err.println("파이썬 스크립트에서 에러발생: " + e.getMessage());
        } catch (InterruptedException e) {
            System.err.println("프로세스 인터럽트 발생 : " + e.getMessage());
        }

    }

    @Transactional(readOnly = true)
    public List<DiaryDTO> getDiaryList(Long memberId) {
        List<Diary> diaryList = diaryRepository.findByMemberId(memberId);
        List<DiaryDTO> diaryDTOList = diaryList
                .stream()
                .map(DiaryDTO::of)
                .toList();

        return diaryDTOList;
    }

    @Transactional(readOnly = true)
    public DiaryDTO getDiary(Long diaryId) {
        Diary diary = diaryRepository.findById(diaryId)
                .orElseThrow();
        return DiaryDTO.of(diary);
    }

    @Transactional
    public void deleteDiary(Long diaryId) {
        diaryRepository.deleteById(diaryId);
    }
}
