package hodu.member.repository;

import hodu.member.domain.Member;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface MemberRepository extends JpaRepository <Member, Long> {
    Optional<Member> findByLoginIdAndPassword(String loginId, String password);
}
