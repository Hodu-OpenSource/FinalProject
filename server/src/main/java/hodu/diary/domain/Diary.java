package hodu.diary.domain;

import hodu.common.model.BaseEntity;
import hodu.member.domain.Member;
import jakarta.persistence.*;

@Entity
public class Diary extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name="member_id")
    private Member member;

    @Column(columnDefinition = "TEXT")
    private String content;

    private String mainEmotion;

    protected Diary() {
    }

    public Diary(Member member, String content, String mainEmotion) {
        this.member = member;
        this.content = content;
        this.mainEmotion = mainEmotion;
    }

    public Long getId() {
        return id;
    }

    public Member getMember() {
        return member;
    }

    public String getContent() {
        return content;
    }

    public String getMainEmotion() {
        return mainEmotion;
    }
}
