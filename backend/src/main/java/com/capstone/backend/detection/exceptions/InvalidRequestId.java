package com.capstone.backend.detection.exceptions;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND)
public class InvalidRequestId extends RuntimeException {
    public InvalidRequestId(String message) {
        super(message);
    }
}

