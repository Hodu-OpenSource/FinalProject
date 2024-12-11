package hodu.diary.repository;

import hodu.diary.domain.Diary;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface DiaryRepository extends JpaRepository<Diary, Long> {
    List<Diary> findByMemberId(Long memberId);


    @Query("SELECT d FROM Diary d WHERE d.member.id = :memberId ORDER BY d.createdDate DESC")
    Optional<Diary> findFirstByMemberIdOrderByCreatedDateDesc(@Param("memberId") Long memberId);

}
