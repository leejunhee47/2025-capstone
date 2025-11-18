package com.capstone.ai.component;

import com.capstone.ai.request.DetectionRequest;
import org.springframework.stereotype.Component;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

@Component
public class AiJobQueue {

    private final BlockingQueue<DetectionRequest> queue = new LinkedBlockingQueue<>();


    // 서비스 호출
    public void addJob(DetectionRequest job) {
        queue.offer(job);
    }

    // Worker 호출
    public DetectionRequest takeJob() {
        return queue.poll();
    }


}
