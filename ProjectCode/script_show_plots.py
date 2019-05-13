# THIS IS A SCRIPT
import os
from visualisations import plot_metrics_from_pkl

usr_folder_path = 'C:/Users/tim-f/Documents/ATML_GroupWork_GitRepo/MLcourseProject/ProjectCode/classification_results/results_to_sync/Best_Model/many_Epochs_longSegment/shorter se'
#usr_folder_path ='C:/Users/tim-f/Documents/ATML_GroupWork_GitRepo/MLcourseProject/ProjectCode/classification_results/results_to_sync/Best_Model/many_Epochs_longSegment'

files_list = []
for file in os.listdir(usr_folder_path):
    if file.endswith("_plt.pkl"):
        files_list.append(os.path.join(usr_folder_path, file))

# Plot them

for tmp_pkl_path in files_list:
    plot_metrics_from_pkl(tmp_pkl_path)