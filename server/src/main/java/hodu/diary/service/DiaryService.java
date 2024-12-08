package hodu.diary.service;

import hodu.diary.domain.Diary;
import hodu.diary.dto.DiaryDTO;
import hodu.diary.repository.DiaryRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DiaryService {
    private final DiaryRepository diaryRepository;

    public DiaryService(DiaryRepository diaryRepository) {
        this.diaryRepository = diaryRepository;
    }

    public void addDiary(Long memberId) {
        //일기 생성 파이썬 코드

       //diaryRepository.save(new Diary()) 일기 저장 코드

    }

    public List<DiaryDTO> getDiaryList(Long memberId) {
        List<Diary> diaryList = diaryRepository.findByMemberId(memberId);
        List<DiaryDTO> diaryDTOList = diaryList
                .stream()
                .map(DiaryDTO::of)
                .toList();

        return diaryDTOList;
    }
}
