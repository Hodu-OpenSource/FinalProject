package hodu.common.exception;

import hodu.member.exception.MemberNotFoundException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

import java.util.HashMap;
import java.util.Map;

@ControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {
    @ExceptionHandler(MemberNotFoundException.class)
    public ResponseEntity<Map<String, String>> HandleMemberNotFoundException(MemberNotFoundException e) {
        Map<String, String> errors = new HashMap<>();
        errors.put("error", e.getMessage());
        return new ResponseEntity<>(errors, HttpStatus.BAD_REQUEST);
    }
}
