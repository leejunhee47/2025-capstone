package com.capstone.backend.member.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import java.time.LocalDateTime;

@Data
@Entity
@NoArgsConstructor
public class Member {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long memberId;

    @Column(unique = true, nullable = false)
    private String username;

    @Column(nullable = false)
    private String password;

    @Column(nullable = false)
    private String name;

    @CreationTimestamp
    @Column(updatable = false , nullable = false)
    private LocalDateTime createdAt;


    public Member(String username, String password, String name) {

        if(username == null || password == null || name == null || username.isEmpty() || password.isEmpty() || name.isEmpty()){
            throw new IllegalArgumentException("null error");
        }

        this.username = username;
        this.password = password;
        this.name = name;
    }

}
