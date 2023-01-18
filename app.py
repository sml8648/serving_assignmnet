import streamlit as st
import requests
import urllib.request

st.set_page_config(layout='wide')

st.title("문장 상관관계 분석")

if "input1" not in st.session_state:
    st.session_state["input1"] = ""

if "input2" not in st.session_state:
    st.session_state["input2"] = ""

with st.form(key="입력 form"):
    sentence1 = st.text_input("sentence1",
                                key="input1",
                                placeholder="문장을 입력하세요")

    sentence2 = st.text_input("sentence2",
                                key="input2",
                                placeholder="문장을 입력하세요")

    submit = st.form_submit_button("문장 제출")

with st.spinner("두뇌 풀가동"):
    if submit:

        data = {
            'sentence1':st.session_state.input1,
            'sentence2':st.session_state.input2
        }

        data = urllib.parse.urlencode(data)
        result = requests.get("http://localhost:30001/inference?", params=data)

        predictions = result.json()['Return_score']

        st.write('분석 결과 : ', round(predictions, 1))

txt = st.text_area('라벨의 의미 (훈련되지 않은 모델임으로 결과가 부정확할 수 있습니다.)', '''
    * Label 점수: 0 ~ 5사이의 실수
    5점 : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함
    4점 : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음
    3점 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음
    2점 : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함
    1점 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음
    0점 : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음
    ''',
    height=240)