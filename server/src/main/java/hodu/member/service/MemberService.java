package hodu.member.service;

import hodu.member.domain.Member;
import hodu.member.exception.MemberNotFoundException;
import hodu.member.repository.MemberRepository;
import org.springframework.stereotype.Service;

@Service
public class MemberService {
    private final MemberRepository memberRepository;

    public MemberService(MemberRepository memberRepository) {
        this.memberRepository = memberRepository;
    }

    public void signUp(String loginId, String password) {
        memberRepository.save(new Member(loginId,password));
    }

    public long login(String loginId, String password) {
        Member member = memberRepository.findByLoginIdAndPassword(loginId, password)
                .orElseThrow(()->new MemberNotFoundException("아이디 또는 비밀번호가 틀렸습니다"));

        return member.getId();

    }
}
