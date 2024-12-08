package hodu.member.controller;

import hodu.member.dto.response.LoginResponse;
import hodu.member.service.MemberService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/member")
public class MemberController {
    private final MemberService memberService;

    public MemberController(MemberService memberService) {
        this.memberService = memberService;
    }

    @PostMapping("/signUp")
    public ResponseEntity<Void> signUp (
            @RequestParam (name="loginId") String loginId,
            @RequestParam (name="password") String password
    ) {
        memberService.signUp(loginId, password);
        return ResponseEntity.status(HttpStatus.CREATED).build();
    }

    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(
            @RequestParam(name = "loginId") String loginId,
            @RequestParam(name = "password") String password
    ) {
        long memberId = memberService.login(loginId, password);
        return ResponseEntity.ok(new LoginResponse(memberId));
    }
}
