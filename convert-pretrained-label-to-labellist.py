import os

if __name__ == '__main__':
    pretrained_labels = []
    label_synset_map = {}

    # Files containing label_name
    with open('pretrained-labels.txt', 'r', encoding='utf-8') as f:
        labels_body = f.read().strip()
        labels_body_list = labels_body.split('\n')
        labels_body_list = [line.strip() for line in labels_body_list]
        pretrained_labels += labels_body_list

    # Files containing n****|label_name
    with open('label.list', 'r', encoding='utf-8') as f:
        labels_body = f.read().strip()
        labels_body_list = labels_body.split('\n')
        labels_body_list = [line.strip() for line in labels_body_list]
        for label in labels_body_list:
            synset_name, label_name = label.split('|')
            label_synset_map[label_name] = synset_name

    # Match item and save as another list
    SAVE_PATH = './pretrained-label.list'
    pretrained_mapped_label = []
    for label in pretrained_labels:
        try:
            synset = label_synset_map[label]
            pretrained_mapped_label.append(synset + '|' + label)
        except Exception:
            print("Specified label is NOT in synset map: ", label)

    with open(SAVE_PATH, 'w', encoding='utf-8') as w:
        for line in pretrained_mapped_label:
            w.write(line + '\n')

    print("Wrote successfully to: ", SAVE_PATH)