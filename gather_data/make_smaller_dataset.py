import os
import shutil

def sort_and_keep_files(folder_path, number_of_files_to_keep):
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Sort files by size in descending order
    all_files.sort(key=lambda x: os.path.getsize(os.path.join(folder_path, x)), reverse=True)

    # Keep only the specified number of largest files
    for file in all_files[number_of_files_to_keep:]:
        os.remove(os.path.join(folder_path, file))

def main():
    base_directory = "small_test_data/data"  # Change this to your base directory

    # Directories for win, draw, lose
    win_directory = os.path.join(base_directory, "win")
    draw_directory = os.path.join(base_directory, "draw")
    lose_directory = os.path.join(base_directory, "lose")

    # Number of files to keep in each directory based on the 10:1:10 ratio
    # Assuming draw has 100 files
    files_in_draw = 100
    files_in_win = files_in_draw * 10
    files_in_lose = files_in_draw * 10

    # Sort and keep files for each folder
    sort_and_keep_files(win_directory, files_in_win)
    sort_and_keep_files(draw_directory, files_in_draw)
    sort_and_keep_files(lose_directory, files_in_lose)

if __name__ == "__main__":
    main()
