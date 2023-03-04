"""
Creation d'un site avec streamlit
"""
from utils import topics_suggestion, positive_review
import streamlit as st
import pandas as pd

st.title("DETECTION DU SUJET D'INSATISFACTION D'UNE ENTREPRISE DE RESTORATION")

def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()


number = st.slider('Choisir un nombre de sujet', value=3, step=1, min_value=1, max_value=15)

with open('../Topic_Modeling/style.css') as f:
    css_component = f'<style>{f.read()}</style>'
    st.markdown(css_component, unsafe_allow_html=True)


st.markdown("<h3> Entrer votre text ci dessous <h3>", unsafe_allow_html=True)
review = st.text_area("✍ Write the opinion", height=200, max_chars=5000, key='options')
detect_topic_btn = st.button(label="✨ Determination des sources d'insatisfaction ")

if detect_topic_btn:
    test = positive_review(review)
    if test:
        st.warning("✔ Ton Opinion est positive veuillez renseignez une nouvelle opinion")
    else:
        suggested_topics = topics_suggestion(review, number)
        print(suggested_topics)
        columns_components = st.columns(len(suggested_topics))
        i = 0
        list1 = []
        list2 = []
        for col in columns_components:
            col.metric(suggested_topics[i][0], suggested_topics[i][1])
            list1.append(suggested_topics[i][0])
            list2.append(float(suggested_topics[i][1].replace("%", "")))
            i += 1
        st.balloons()

        "Probability par sujet"
        source = pd.DataFrame({
            'Probabilité': list2,
            'SUJET': list1
        })
        import altair as alt

        bar_chart = alt.Chart(source).mark_bar(color="#DE3163").encode(
            y='Probabilité:Q',
            x='SUJET:O',
        )
        st.altair_chart(bar_chart, use_container_width=True)

        if len(suggested_topics) != number:
            st.warning(
                "Le nombre de topic que vous avez demandé est supérieur au nombre de topic "
                "qui peuvent être en relation avec ce review (Probabilité de similarité égale à 0%)"
            )

format_dictionary = {
    "Relevancy": "{:.1%}",
}
