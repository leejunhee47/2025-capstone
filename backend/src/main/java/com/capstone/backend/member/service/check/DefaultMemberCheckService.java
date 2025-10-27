//package com.capstone.backend.member.service.check;
//
//import bankapp.account.model.account.Account;
//import bankapp.member.exceptions.MemberNotFoundException;
//import bankapp.member.model.Member;
//import bankapp.member.repository.MemberRepository;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.stereotype.Service;
//
//
//// TODO: 테스트 코드 작성
///**
// * {@inheritDoc}
// * <p>
// * 이 구현체는 MemberDao를 사용하여 데이터베이스를 체크하여
// * 회원 정보 확인 및 조회 로직을 수행합니다.
// */
//@Service
//public class DefaultMemberCheckService implements MemberCheckService {
//
//    private final MemberRepository memberRepository;
//
//    @Autowired
//    public DefaultMemberCheckService(MemberRepository memberRepository){
//        this.memberRepository = memberRepository;
//    }
//
//
//    @Override
//    public boolean isMemberIdExist(Long memberId){
//        if(memberId == null) {
//            return false;
//        }
//
//        return memberRepository.existsById(memberId);
//    }
//
//    @Override
//    public boolean isUsernameExist(String username){
//        if(username == null) {
//            return false;
//        }
//
//        return memberRepository.existsByUsername(username);
//    }
//
//    @Override
//    public Member findMemberByAccount(Account account) throws MemberNotFoundException{
//        if(account == null || account.getMember() == null) {
//            throw new MemberNotFoundException("멤버를 찾을 수 없습니다.");
//        }
//
//        return account.getMember();
//    }
//
//
//}
