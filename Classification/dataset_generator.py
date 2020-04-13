import pandas as pd
import numpy as np

# Take the class data from the excel file and create a dataset from it
def generate_dataset(sample_size, offset_range):
    # sample_size => number of sets of data to make for each class
    # offset_range => added noise to the orginal class data [-, +] (percentage)
    dataset = np.empty((0, 46))
    for row in range(4):
        y = template_data[row][0]
        x = template_data[row][1:]
        x = x.astype(float)
        for i in range(sample_size):
            offset = np.random.uniform(low=-offset_range, high=offset_range, size=[45,])
            offset = np.multiply(x, offset)
            new_x = np.add(x, offset)
            new_row = np.append(new_x, y)
            dataset = np.append(dataset, [new_row], axis=0)
    
    return dataset

excel_data = pd.read_excel('Take home exam dataset.xlsx', skiprows=[0], usecols=range(1, 47))
template_data = excel_data.to_numpy()
dataset = generate_dataset(sample_size=20, offset_range=0.1)
pd.DataFrame(dataset).to_csv('dataset.csv', header=None, index=None)
print('Save dataset to: dataset.csv')