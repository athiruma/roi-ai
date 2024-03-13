import os
from typing import Any
from google.auth.transport.requests import Request

from google_auth_oauthlib.flow import InstalledAppFlow
from llama_index.readers.google import GoogleDocsReader
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]


class CustomGoogleReader(GoogleDocsReader):

    def __init__(self):
        super().__init__()

    def _get_credentials(self) -> Any:

        creds = None
        if os.path.exists(".token.json"):
            creds = Credentials.from_authorized_user_file(".token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=51153)
            # Save the credentials for the next run
            with open(".token.json", "w") as token:
                token.write(creds.to_json())
