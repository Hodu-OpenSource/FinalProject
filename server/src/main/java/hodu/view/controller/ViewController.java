package hodu.view.controller;

import hodu.diary.dto.DiaryDTO;
import hodu.diary.service.DiaryService;
import hodu.member.service.MemberService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

import java.util.List;

@Controller
@RequestMapping("/view/")
public class ViewController {

    private final DiaryService diaryService;
    private final MemberService memberService;

    public ViewController(DiaryService diaryService, MemberService memberService) {
        this.diaryService = diaryService;
        this.memberService = memberService;
    }

    //최초 메인페이지
    @GetMapping("/mainPage")
    public String mainPage () {
        return "mainPage";
    }

    //로그인 페이지
    @GetMapping("/loginPage")
    public String loginPage() {
        return "loginPage";
    }

    //회원가입 페이지
    @GetMapping("/signUpPage")
    public String signUpPage() {
        return "signUpPage";
    }

    //일기 메인페이지
    @GetMapping("/diaryMainPage/{memberId}")
    public String diaryMainPage(
            @PathVariable("memberId") Long memberId,
            Model model
    ){
        List<DiaryDTO> diaryList = diaryService.getDiaryList(memberId);
        boolean isDiaryWrittenToday = memberService.getIsDiaryWrittenToday(memberId);

        model.addAttribute("diaryList", diaryList);
        model.addAttribute("isDiaryWrittenToday", isDiaryWrittenToday);

        return "diaryMainPage";
    }

    //상세 일기 페이지
    @GetMapping("/diaryDetail/{diaryId}")
    public String diaryDetail (
            @PathVariable("diaryId") Long diaryId,
            Model model
    ) {
        DiaryDTO diary = diaryService.getDiary(diaryId);
        model.addAttribute("diary", diary);
        return "diaryDetailPage";
    }

    @GetMapping("/writeDiaryPage")
    public String writeDiary(
    ) {
        return "writeDiaryPage";
    }
}
