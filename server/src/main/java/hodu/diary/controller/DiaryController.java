package hodu.diary.controller;

import hodu.diary.dto.DiaryDTO;
import hodu.diary.service.DiaryService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/diary")
public class DiaryController {
    private final DiaryService diaryService;

    public DiaryController(DiaryService diaryService) {
        this.diaryService = diaryService;
    }

    @PostMapping("/{memberId}")
    public ResponseEntity<Void> addDiary(
            @PathVariable("memberId") Long memberId
    ) {
        diaryService.addDiary(memberId);
        return ResponseEntity.status(HttpStatus.CREATED).build();
    }

    @DeleteMapping("/{diaryId}")
    public ResponseEntity<Void> deleteDiary(
            @PathVariable("diaryId") Long diaryId
    ) {
        diaryService.deleteDiary(diaryId);
        return ResponseEntity.status(HttpStatus.OK).build();
    }
}
