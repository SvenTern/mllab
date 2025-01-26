import os
import json
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm


class GoogleDriveHandler:
    def __init__(self, credentials_path='credentials.json', token_path='token.json', scopes=None):
        """Initializes Google Drive handler and authenticates the application."""
        self.SCOPES = scopes if scopes else ['https://www.googleapis.com/auth/drive']
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds = None

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Authenticate the service
        self.service = self.authenticate()

    def authenticate(self):
        """Handles authentication to Google Drive API."""
        try:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError("credentials.json not found!")

            if os.path.exists(self.token_path):
                self.creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)

            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
                    self.creds = flow.run_local_server(port=0)

                with open(self.token_path, 'w') as token:
                    token.write(self.creds.to_json())

            return build('drive', 'v3', credentials=self.creds)

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise

    def get_folder_id(self, folder_name, parent_id="root"):
        """Gets the ID of a folder by name starting from a specific parent."""
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            folders = results.get('files', [])
            if folders:
                return folders[0]['id']
            else:
                raise FileNotFoundError(f"Folder '{folder_name}' not found in parent '{parent_id}'.")
        except Exception as e:
            self.logger.error(f"Error finding folder {folder_name}: {e}")
            raise

    def list_files_in_folder(self, folder_id):
        """Lists files and folders in a specific folder."""
        try:
            results = self.service.files().list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType, size)"
            ).execute()
            return results.get('files', [])
        except Exception as e:
            self.logger.error(f"Error listing files in folder {folder_id}: {e}")
            raise

    def count_all_files(self, folder_id):
        """Counts all files recursively in a folder."""
        total_files = 0
        files = self.list_files_in_folder(folder_id)
        for file in files:
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                total_files += self.count_all_files(file['id'])
            else:
                total_files += 1
        return total_files

    def file_exists_locally(self, local_path, drive_file_size):
        """Checks if a file exists locally and matches the size on Google Drive."""
        return os.path.exists(local_path) and os.path.getsize(local_path) == int(drive_file_size)

    def download_file(self, file_id, file_name, local_folder, drive_file_size, progress_bar):
        """Downloads a file from Google Drive to a local folder if it doesn't already exist."""
        local_path = os.path.join(local_folder, file_name)
        if self.file_exists_locally(local_path, drive_file_size):
            #progress_bar.write(f"File already exists and matches size: {file_name}. Skipping download.")
            return
        try:
            request = self.service.files().get_media(fileId=file_id)
            with open(local_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            #progress_bar.write(f"Downloaded {file_name} to {local_path}")
        except Exception as e:
            progress_bar.write(f"Failed to download {file_name}: {e}")
            raise

    def download_folder_with_progress(self, folder_id, local_folder):
        """
        Downloads an entire folder and its contents from Google Drive with a global progress bar.
        """
        total_files = self.count_all_files(folder_id)
        with tqdm(total=total_files, desc="Copying all files", unit="file", leave=True) as global_progress:
            self._download_folder_recursive(folder_id, local_folder, global_progress)

    def _download_folder_recursive(self, folder_id, local_folder, progress_bar):
        """Downloads all files and folders recursively from a Google Drive folder."""
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        try:
            files = self.list_files_in_folder(folder_id)
            for file in files:
                if file['mimeType'] == 'application/vnd.google-apps.folder':  # It's a folder
                    subfolder_path = os.path.join(local_folder, file['name'])
                    os.makedirs(subfolder_path, exist_ok=True)
                    self._download_folder_recursive(file['id'], subfolder_path, progress_bar)
                else:  # It's a file
                    self.download_file(file['id'], file['name'], local_folder, file.get('size', 0), progress_bar)
                progress_bar.update(1)  # Update the global progress bar
        except Exception as e:
            progress_bar.write(f"Error downloading folder recursively: {e}")
            raise


# Основной скрипт
if __name__ == "__main__":
    drive = GoogleDriveHandler(credentials_path='credentials.json', token_path='token.json')

    try:
        # Подключение к корневой директории "Мой диск"
        root_folder_id = "root"

        # Найти папку Data_trading/SP500_1m
        data_trading_id = drive.get_folder_id("Data_trading", parent_id=root_folder_id)
        sp500_folder_id = drive.get_folder_id("SP500_1m", parent_id=data_trading_id)

        # Локальная папка назначения
        local_data_folder = "data"

        # Скачивание папки с прогресс-баром
        drive.download_folder_with_progress(sp500_folder_id, local_data_folder)

        print(f"Все данные успешно загружены в папку {local_data_folder}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
