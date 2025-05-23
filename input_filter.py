# input_filter.py

import re

class InputFilter:
    """
    사용자 입력 텍스트에 PII 등 민감 정보가 포함되어 있는지
    정규표현식 기반으로 검사합니다.
    """

    # 검출할 패턴들: 필요에 따라 추가/수정하세요.
    patterns = [
        # 주민등록번호 (6자리-7자리)
        re.compile(r"\b\d{6}-\d{7}\b"),
        # 주민등록번호 (13자리 숫자) ※ 하이픈 없이
        re.compile(r"\b\d{13}\b"),
        # 전화번호 (국내) 예: 010-1234-5678, 02-123-4567
        re.compile(r"\b\d{2,4}-\d{3,4}-\d{4}\b"),
        # 이메일 주소
        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        # 신용카드 번호 (예시, 카드사별 포맷 추가 가능)
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    ]

    @classmethod
    def contains_sensitive(cls, text: str) -> bool:
        """
        text 안에 정의된 패턴 중 하나라도 매칭되면 True를 반환합니다.
        """
        for pat in cls.patterns:
            if pat.search(text):
                return True
        return False
