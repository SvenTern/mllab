import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import os

class GoogleDriveHandler:
    def __init__(self, credentials_path='credentials.json', token_path='token.json', scopes=None):
        """Initializes Google Drive handler and authenticates the application."""
        self.SCOPES = scopes if scopes else ['https://www.googleapis.com/auth/drive']
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds = None
        self.service = self.authenticate()

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def authenticate(self):
        """Handles authentication to Google Drive API."""
        try:
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

    def download_file(self, file_id, destination_path):
        """Downloads a file from Google Drive."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            with open(destination_path, "wb") as local_file:
                downloader = MediaIoBaseDownload(local_file, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    self.logger.info(f"Download progress: {int(status.progress() * 100)}%")
            self.logger.info(f"File downloaded successfully to {destination_path}")
        except HttpError as e:
            self.logger.error(f"Error downloading file: {e}")
            raise

    def upload_file(self, file_path, folder_id=None):
        """Uploads a file to Google Drive."""
        try:
            file_metadata = {'name': os.path.basename(file_path)}
            if folder_id:
                file_metadata['parents'] = [folder_id]

            media = MediaFileUpload(file_path, resumable=True)
            response = self.service.files().create(body=file_metadata, media_body=media, fields="id").execute()
            self.logger.info(f"File uploaded successfully: {response.get('id')}")
            return response.get('id')
        except HttpError as e:
            self.logger.error(f"Error uploading file: {e}")
            raise

    def list_files(self, folder_id=None, query=None):
        """Lists files in Google Drive."""
        try:
            query = query or ""
            if folder_id:
                query += f"'{folder_id}' in parents"

            results = self.service.files().list(q=query, pageSize=100, fields="files(id, name)").execute()
            files = results.get('files', [])
            for file in files:
                self.logger.info(f"Found file: {file['name']} ({file['id']})")
            return files
        except HttpError as e:
            self.logger.error(f"Error listing files: {e}")
            raise

    def file_exists(self, file_name, folder_id=None):
        """Checks if a file with the specified name exists in Google Drive."""
        try:
            query = f"name = '{file_name}'"
            if folder_id:
                query += f" and '{folder_id}' in parents"

            results = self.service.files().list(q=query, pageSize=1, fields="files(id, name)").execute()
            files = results.get('files', [])
            return files[0] if files else None
        except HttpError as e:
            self.logger.error(f"Error checking file existence: {e}")
            raise

    def upload_file_with_check(self, file_path, folder_id=None):
        """Uploads a file to Google Drive, checking for duplicates."""
        file_name = os.path.basename(file_path)
        existing_file = self.file_exists(file_name, folder_id)
        if existing_file:
            self.logger.warning(f"File '{file_name}' already exists with ID: {existing_file['id']}")
            return existing_file['id']
        return self.upload_file(file_path, folder_id)

    def handle_rate_limiting(self):
        """Handles rate-limiting logic (placeholder for implementation)."""
        self.logger.warning("Rate-limiting handling is not implemented.")
        # Logic for handling rate limits can be added here.

# Usage example
# drive_handler = GoogleDriveHandler()
# drive_handler.download_file(file_id="your_file_id", destination_path="your_path")
