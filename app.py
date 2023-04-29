# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model


def main():

	st.set_page_config(page_title="My app s20777")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://image.freepik.com/darmowe-wektory/choroba-serca-edukacja-miazdzycowa-medycyna_68292-824.jpg")

	with overview:
		st.title("My app s20777")

	# with left:
	# 	sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
	# 	pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x : pclass_d[x])
	# 	embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )


	with left:
		age_slider = st.slider("wiek", value=1, min_value=1, max_value=90)
		sibsp_slider = st.slider("Chorby współistniejace ", min_value=0, max_value=10)
		parch_slider = st.slider("Wzrost", min_value=0, max_value=200)
		fare_slider = st.slider("Leki", min_value=0, max_value=420, step=1)

	data = [[age_slider, sibsp_slider, parch_slider, fare_slider, fare_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba bedzie miec chorobe serca?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
