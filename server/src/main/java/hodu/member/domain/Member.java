package hodu.member.domain;

import hodu.common.model.BaseEntity;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Member extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String loginId;

    private String password;

    private long attendanceCount;

    private boolean isDiaryWrittenToday;

    protected Member() {
    }

    public Member(String loginId, String password) {
        this.attendanceCount = 0L;
        this.password = password;
        this.loginId = loginId;
        this.isDiaryWrittenToday = false;
    }

    public Long getId() {
        return id;
    }

    public String getLoginId() {
        return loginId;
    }

    public String getPassword() {
        return password;
    }

    public long getAttendanceCount() {
        return attendanceCount;
    }

    public boolean getIsDiaryWrittenToday() {
        return isDiaryWrittenToday;
    }

    public void setDiaryWrittenToday() {
        this.isDiaryWrittenToday = true;
    }

    public void resetIsDiaryWrittenTodayStatus() {
        this.isDiaryWrittenToday = false;
    }
}
