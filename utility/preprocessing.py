from sklearn.preprocessing import LabelEncoder


def prepare_data(data, text_col, target_col, initial_target):
    data[initial_target] = data[initial_target].replace(regex="(@\w+)|#|&|!", value="")
    data[initial_target] = data[initial_target].replace(regex=r"http\S+", value="")
    data[initial_target] = data[initial_target].replace(regex=r"RT", value="")
    data[initial_target] = data[initial_target].replace(regex=r":", value="")

    text_classes = data[initial_target].unique()
    label_encoder = LabelEncoder()
    int_classes = label_encoder.fit_transform(text_classes)
    data_category_map = dict(zip(text_classes, int_classes))
    data[target_col] = [data_category_map[label] for label in data[initial_target]]

    return (data, data_category_map)
