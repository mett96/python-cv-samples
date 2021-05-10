import os

folders = ["gauge_coco_backgrounds", "gauges_random_backgrounds"]
starting_folder = folders[1]

folder = os.path.join("images", starting_folder)

if __name__ == '__main__':
    for i, file_name in enumerate(sorted(os.listdir(folder))):
        new_filename = f"gauge-{i}.jpg"
        os.rename(os.path.join(folder, file_name), os.path.join(folder, new_filename))
