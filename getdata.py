import os
import shutil

# --- CONFIGURATION ---
# The path you gave me
SOURCE_ROOT = r"C:\Users\myltb\Downloads\POP909"

# Where you want the files to go (creates a new folder next to the source)
DESTINATION_DIR = r"C:\Users\myltb\Downloads\POP909_All_Midi"

def extract_midis():
    # 1. Create destination folder if it doesn't exist
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
        print(f"Created destination folder: {DESTINATION_DIR}")

    total_files_copied = 0

    # 2. Loop through folders 001 to 909
    for i in range(1, 910):
        # Format number to 3 digits (e.g., 1 -> "001")
        folder_name = f"{i:03d}"
        
        # Path to the current numbered folder
        current_folder_path = os.path.join(SOURCE_ROOT, folder_name)
        
        # Skip if for some reason the folder doesn't exist
        if not os.path.exists(current_folder_path):
            continue

        # --- A. GRAB THE MAIN FILE (e.g., 001.mid) ---
        # User specified the file is named "001.mid" inside the "001" folder
        main_midi_name = f"{folder_name}.mid"
        main_midi_path = os.path.join(current_folder_path, main_midi_name)

        if os.path.exists(main_midi_path):
            # New name: 001_main.mid
            new_name = f"{folder_name}_main.mid"
            target_path = os.path.join(DESTINATION_DIR, new_name)
            
            try:
                shutil.copy2(main_midi_path, target_path)
                total_files_copied += 1
            except Exception as e:
                print(f"Error copying {main_midi_name}: {e}")

        # --- B. GRAB THE VERSIONS (e.g., versions/*.mid) ---
        versions_folder = os.path.join(current_folder_path, "versions")
        
        if os.path.exists(versions_folder):
            # Loop through all files in the versions folder
            for version_file in os.listdir(versions_folder):
                if version_file.endswith(".mid") or version_file.endswith(".midi"):
                    source_version_path = os.path.join(versions_folder, version_file)
                    
                    # New name: 001_v_filename.mid (prevents duplicates)
                    new_version_name = f"{folder_name}_v_{version_file}"
                    target_version_path = os.path.join(DESTINATION_DIR, new_version_name)
                    
                    try:
                        shutil.copy2(source_version_path, target_version_path)
                        total_files_copied += 1
                    except Exception as e:
                        print(f"Error copying version {version_file}: {e}")

        # Optional: Print progress every 100 folders
        if i % 100 == 0:
            print(f"Processed up to folder {folder_name}...")

    print("-" * 30)
    print(f"DONE! Successfully extracted {total_files_copied} MIDI files.")
    print(f"They are located in: {DESTINATION_DIR}")

if __name__ == "__main__":
    extract_midis()