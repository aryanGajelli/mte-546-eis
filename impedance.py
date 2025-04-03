from read_bin import load_data
from data_cleaner import break_data, break_hf

df, fs = load_data(cell=79, temp='35C', soc=0)
lf_v, hf_v = break_data(df, volt=True)
lf_i, hf_i = break_data(df, volt=False)