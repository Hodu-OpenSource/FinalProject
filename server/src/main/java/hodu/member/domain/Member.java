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

    protected Member() {
    }

    public Member(String loginId, String password) {
        this.attendanceCount = 0L;
        this.password = password;
        this.loginId = loginId;
    }

    public void addAttendanceCount () {
        attendanceCount++;
    }
}
