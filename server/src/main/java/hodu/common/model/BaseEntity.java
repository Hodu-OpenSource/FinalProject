package hodu.common.model;

import jakarta.persistence.Column;
import jakarta.persistence.EntityListeners;
import jakarta.persistence.MappedSuperclass;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.sql.Timestamp;

@MappedSuperclass
@EntityListeners(value = {AuditingEntityListener.class})
public class BaseEntity {
    @CreatedDate
    @Column(nullable = false, updatable = false)
    private Timestamp createdDate;

    public Timestamp getCreatedDate() {
        return createdDate;
    }
}
