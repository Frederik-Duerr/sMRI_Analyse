import pathlib 
import os




Data_folder = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI', 'SMC']


def read_folders(main_path):
    image_data =[]
    for root, dirs, files in os.walk(main_path):

        for file in files:

            if file.lower().endswith(".png"):

                file_path = os.path.join(root, file)

                # Extract information from the folder structure
                relative_path = os.path.relpath(file_path, main_path)
                path_parts = relative_path.split(os.sep)

                diagnosis = path_parts[0]
                patient_id = path_parts[1]

                image_data.append({
                    "image_path": file_path,
                    "diagnosis": diagnosis,
                    "patient_id": patient_id,
                    "filename": file
                })
    return image_data


#print(meta_data_frame.sample(10))
#print(meta_data_frame.info())

def sclicing_image_features(meta_data_frame):
    sclices = ['axial','coronal','sagittal']
    # print(type(data_frame_file_name)) # Its a series
    orientation = []
    Sclice = []
    Sclice_Number =[]
    for i in meta_data_frame['filename']:

        # print(i.split("_")) # generates a list of [id, orientation, sclice, number]
        orientation.append(i.split("_")[1])
        #print(i.split("_")[1])
        Sclice.append( i.split("_")[2])
        #print(i.split("_")[2])
        split_i = i.split("_")
        #print(split_i)
        sclice_number = split_i[3].split(".")
        #print(sclice_number)
        Sclice_Number.append(sclice_number[0])
    meta_data_frame['Orientation'] = orientation
    meta_data_frame['Sclice'] = Sclice
    meta_data_frame['Sclice_Number'] = Sclice_Number
    return meta_data_frame
