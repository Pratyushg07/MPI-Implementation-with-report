import pickle

# Path to your pickle file
pickle_file_path = "path-to-save.pickle"

# Open the pickle file in read mode
with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)

# Print the loaded data
print(data)
