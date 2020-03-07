import numpy as np
import pandas as pd
import glob




if __name__ == "__main__":
	
	flag = "val"
	file_list = glob.glob("../../data/augmented_data/"+str(flag)+"/wav_files/*")
	total_list = []
	for current_file_path in file_list:
		print(current_file_path)
		current_file_id = current_file_path.split("/")[6].split(".")[0]
		current_class = current_file_id.split("_")[1]
		total_list.append([current_file_id,current_class])


	columns = ['file_id',"label"]
	print("[INFO] SAVING DATA IN DATAFRAME")
	df = pd.DataFrame(index=np.arange(len(total_list)), columns=columns)
	df.loc[:] = total_list
	df.to_csv("../../data/augmented_data/"+str(flag)+"/"+str(flag)+".csv", encoding="utf-8", index=False)
	



