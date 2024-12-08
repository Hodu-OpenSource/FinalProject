package hodu.diary.dto;

import hodu.diary.domain.Diary;

import java.time.format.DateTimeFormatter;

public record DiaryDTO (
        Long id,
        String content,
        String mainEmotion,
        String createdDate

) {
    public static DiaryDTO of (Diary diary) {
        DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        return new DiaryDTO(diary.getId(),
                diary.getContent(),
                diary.getMainEmotion(),
                diary.getCreatedDate().toLocalDateTime().toLocalDate().format(dateTimeFormatter));
    }
}
