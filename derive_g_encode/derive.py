import pickle


def read_pickle(name):
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)


# read_file = "499_pred_GDA_new.pkl"
read_file = "499_pred.pkl"
num_domain = 60

info = read_pickle(read_file)
z = info["z"]

# print(z)

g_encode = dict()
for i in range(num_domain):
    g_encode[str(i)] = z[i]

write_pickle(g_encode, "g_encode_60.pkl")
print("success!")
