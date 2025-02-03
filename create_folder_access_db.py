import sqlite3
import os

# Define folder names and their corresponding access levels
folders = {
    "common_base": ["admin", "sales", "hr", "it"],
    "dept/hr": ["hr"],  # hr is now a subfolder under dept
    "dept/it": ["it"],  # it is now a subfolder under dept
    "dept/sales": ["sales"]  # sales is now a subfolder under dept
}

# SQLite database setup
db_file = 'documents_access.db'

def create_database():
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create a table to store files and their access levels
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            folder_name TEXT NOT NULL,
            access_level TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def insert_file(file_name, folder_name, access_level):
    # Insert a file record with its access level
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO files (file_name, folder_name, access_level)
        VALUES (?, ?, ?)
    ''', (file_name, folder_name, access_level))

    conn.commit()
    conn.close()


def populate_files_from_folders(base_directory):
    for folder, access_levels in folders.items():
        folder_path = os.path.join(base_directory, folder)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file_name)):
                    access_level = ','.join(access_levels)
                    insert_file(file_name, folder, access_level)
                    print(f'Inserted file: {file_name} from folder: {folder} with access level: {access_level}')
        else:
            print(f"Folder {folder_path} does not exist.")


def get_all_files():
    # Fetch all files and their access levels
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM files")
    files = cursor.fetchall()

    conn.close()
    return files


def get_files_for_role(role):
    # Fetch files that are accessible to a specific role
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT file_name, folder_name FROM files
        WHERE access_level LIKE ? OR access_level LIKE '%common%'
    ''', (f'%{role}%', ))

    files = cursor.fetchall()
    conn.close()
    return files


if __name__ == '__main__':
    # Set up the database and insert files from folders
    base_directory = './data'  # Change to the path where your folders are stored

    create_database()
    populate_files_from_folders(base_directory)

    # Optional: Print all files in the database
    all_files = get_all_files()
    print("\nAll files in the database:")
    for file in all_files:
        print(file)

    # Test: Fetch files for a specific role (e.g., "admin")
    role = 'admin'
    admin_files = get_files_for_role(role)
    print(f"\nFiles accessible to {role}:")
    for file in admin_files:
        print(file)
