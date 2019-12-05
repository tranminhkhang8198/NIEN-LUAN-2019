import string


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, "r")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_descriptions(doc):
    descriptions = dict()
    for line in doc.split("\n"):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as image id, the rest as description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split(".")[0]
        # convert description tokens back to string
        image_desc = " ".join(image_desc)
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)
    return descriptions


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans("", "", string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = " ".join(desc)


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + " " + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


# Preprocess and save descriptions to disk
filename = "./dataset_text/token.txt"
doc = load_doc(filename)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print("Original Vocabulary Size: %d" % len(vocabulary))

save_descriptions(descriptions, "descriptions.txt")
