st.subheader("Download Sample Files")

# Sample 1 - Normal
normal_sample = """value
25.1
25.2
24.9
25.0
25.3
25.1
25.2
25.0
"""

st.download_button(
    label="Download Normal Sample",
    data=normal_sample,
    file_name="normal_sample.csv",
    mime="text/csv"
)

# Sample 2 - Noise
noise_sample = """value
25
40
10
35
15
30
20
45
12
38
18
33
"""

st.download_button(
    label="Download Noise Sample",
    data=noise_sample,
    file_name="noise_sample.csv",
    mime="text/csv"
)