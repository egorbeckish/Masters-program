import streamlit as st
import os
import pydicom
import matplotlib.pyplot as plt


path = "DICOM"
slider = st.slider(
	"Slider",
	0,
	len(os.listdir(path))
)

dicom_file = os.listdir(path)[slider]
fig, ax = plt.subplots()
try:
	bytes_file = pydicom.dcmread(f"{path}/{dicom_file}").pixel_array
	ax.imshow(bytes_file, plt.cm.gray)
	st.pyplot(fig)
except:
	ax.plot()
	st.pyplot(fig)