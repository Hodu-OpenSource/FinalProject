package hodu.view.controller;

import hodu.diary.dto.DiaryDTO;
import hodu.diary.service.DiaryService;
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

    public ViewController(DiaryService diaryService) {
        this.diaryService = diaryService;
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

        model.addAttribute("diaryList", diaryList);

        return "diaryMainPage";
    }

}
