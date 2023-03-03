import json


def replace_null_with_0(data):
    """
    Recursively replaces all null values with 0 in a nested dictionary.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                replace_null_with_0(value)
            elif isinstance(value, list):
                for item in value:
                    replace_null_with_0(item)
            elif key == "mask_fill_value" and value is None:
                data[key] = 0
    elif isinstance(data, list):
        for item in data:
            replace_null_with_0(item)


# Open the JSON file for reading
with open("epoch_46.json", "r") as file:
    # Load the JSON data into a Python object
    data = json.load(file)

    # Replace all null values with 0
    replace_null_with_0(data)

# Write the updated data back to the JSON file
with open("epoch_46_mod.json", "w") as file:
    json.dump(data, file)
