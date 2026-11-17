import csv
import pickle

def write_dicts_to_csv(dict_list, filename):
    if not dict_list:
        print("The list of dictionaries is empty. No data to write.")
        return

    # Extract fieldnames and types
    fieldnames = dict_list[0].keys()
    types = {field: type(value).__name__ for field, value in dict_list[0].items()}

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the types row
        writer.writerow(types)

        # Write the data rows
        writer.writerows(dict_list)
    print(f'Data has been written to {filename}')

def read_dict_from_pkl(read_filename):
    with open(read_filename, 'rb') as file:
        read_data = pickle.load(file)
    return read_data

def write_dict_to_pkl(write_dict, write_filename):
    with open(write_filename, 'wb') as file:
        pickle.dump(write_dict, file)
    print(f'Data has been written to {write_filename}')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)