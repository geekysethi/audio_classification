import numpy as np
import pandas as pd
import shutil
import glob
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
	file_path = "/home/ashish18024/Deep-Learning-Assignments/assignment_2/question_3/data/UrbanSound8K/metadata/UrbanSound8K.csv"
	df = pd.read_csv(file_path)

	df["slice_file_name"] =  df['slice_file_name'].str.split('.',expand=True)[0]
	columns = ['file_id',"label"]
	new_df = pd.DataFrame(index=np.arange(len(df)), columns=columns)
	new_df["file_id"] = df["slice_file_name"]
	new_df["label"] = df["classID"]
	train_df,val_df = train_test_split(new_df, test_size=0.20, random_state=42)


	print("[INFO] SAVING DATA IN DATAFRAME")
	train_df.to_csv("../../data/UrbanSound8K/train.csv", encoding="utf-8", index=False)
	val_df.to_csv("../../data/UrbanSound8K/val.csv", encoding="utf-8", index=False)
