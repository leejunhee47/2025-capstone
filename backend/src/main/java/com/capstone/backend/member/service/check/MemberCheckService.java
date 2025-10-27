//package com.capstone.backend.member.service.check;
//
//import bankapp.account.model.account.Account;
//import bankapp.member.exceptions.MemberNotFoundException;
//import bankapp.member.model.Member;
//
///**
// * 회원 정보의 존재 여부를 확인하거나, 특정 조건에 맞는 회원을 조회하는 비즈니스 로직 명세를 정의
// * 주로 다른 서비스에서 유효성 검증 목적으로 사용
// */
//public interface MemberCheckService {
//
//    /**
//     * 주어진 회원 ID가 실제 데이터베이스에 존재하는지 확인합니다.
//     *
//     * @param memberId 존재 여부를 확인할 회원의 고유 ID
//     * @return 회원이 존재하면 true, 그렇지 않으면 false를 반환합니다.
//     */
//    boolean isMemberIdExist(Long memberId);
//
//    /**
//     * 주어진 사용자 아이디(username)가 이미 사용 중인지 확인합니다.
//     *
//     * @param username 존재 여부를 확인할 사용자의 아이디
//     * @return 아이디가 존재하면 true, 그렇지 않으면 false를 반환합니다.
//     */
//    boolean isUsernameExist(String username);
//
//    /**
//     * 주어진 계좌(Account) 객체를 소유한 회원(Member) 정보를 조회합니다.
//     * 계좌 객체에 포함된 회원 ID를 사용하여 회원을 찾습니다.
//     *
//     * @param account 소유자를 찾을 대상 계좌 객체
//     * @return 계좌를 소유한 회원 엔티티
//     * @throws MemberNotFoundException 계좌 정보에 해당하는 회원을 찾을 수 없을 경우 발생 (데이터 정합성 오류)
//     */
//    Member findMemberByAccount(Account account) throws MemberNotFoundException;
//
//
//
//
//}
