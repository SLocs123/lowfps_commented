from test_paths import build_test_paths, get_path


def read_labels(file_path):
    labels = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # [frame_idx, track_id, x, y, w, h, score, cls] = parts --- IGNORE ---
            if len(parts) >= 8:  # Ensure there are enough parts to read the class label
                cls = int(float(parts[7]))  # Class label is the 8th element (index 7)
                id = int(parts[1])  # Track ID is the 2nd element (index 1)
                labels[id] = cls  # Keep the last class seen for each track ID.
    return labels

def read_direction_labels(file_path):
    directions ={}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == "frame":  # Skip header line
                continue
            
            id = int(float(parts[1]))  # Track ID is the 2nd element (index 1)
            direction = parts[4]  # Direction label is the 5th element (index 4)
            if id not in directions:  # Only consider the first occurrence of each track ID
                directions[id] = direction
    return directions

def count_labels(label, direction_label):
    label_map = {0: "person",
                 1: "bicycle",
                 2: "car",
                 3: "motorcycle",
                 4: "airplane",
                 5: "bus",
                 6: "train",
                 7: "truck",
                 8: "boat",
                 9: "traffic light"}
    
    classes = read_labels(label)
    lanes = read_direction_labels(direction_label)
    counts_channel_1 = {label_map[i]: 0 for i in range(10)}
    counts_channel_2 = {label_map[i]: 0 for i in range(10)}
    total_1 = 0
    total_2 = 0
    total = 0
    for id, cls in classes.items():
        if cls in label_map:
            direction = lanes.get(id, "unknown")
            if direction == "LtR":
                counts_channel_1[label_map[cls]] += 1
                total_1 += 1
            elif direction == "RtL":
                counts_channel_2[label_map[cls]] += 1
                total_2 += 1
            total += 1            
    counts_channel_1["total_1"] = total_1
    counts_channel_2["total_2"] = total_2
    return counts_channel_1, counts_channel_2, total     

if __name__ == "__main__":
    test_name = "demo_count"
    paths = build_test_paths(test_name)
    track_labels_path = get_path(paths, "tracks")
    direction_labels_path = get_path(paths, "direction")

    of_interest = ["car", "motorcycle", "bus", "truck", "total_1", "total_2", "total"]
    label_counts_1, label_counts_2, total = count_labels(track_labels_path, direction_labels_path)
    print(f"Total tracks: {total}")
    print("\nLane 1 (LtR) counts:")
    for label, count in label_counts_1.items():
        if label in of_interest:
            print(f"{label}: {count}")
    print("\nLane 2 (RtL) counts:")
    for label, count in label_counts_2.items():
        if label in of_interest:
            print(f"{label}: {count}")