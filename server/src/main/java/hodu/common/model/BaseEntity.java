package hodu.common.model;

import jakarta.persistence.Column;
import org.springframework.data.annotation.CreatedDate;

import java.sql.Timestamp;

public class BaseEntity {
    @CreatedDate
    @Column(nullable = false, updatable = false)
    private Timestamp createdDate;
}
