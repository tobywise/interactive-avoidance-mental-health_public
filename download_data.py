import requests
import zipfile
import io
import os
import sys

# URL of the zip file (hardcoded)
zip_url = "https://files.de-1.osf.io/v1/resources/a7e3k/providers/osfstorage/?zip=&_gl=1*w0a105*_ga*NjI4MjUwODcwLjE2NDgwNjk5NDQ.*_ga_YE9BMGGWX8*MTcwNDIxMzQxMS4zNS4xLjE3MDQyMTU3MzIuNjAuMC4w"

def download_and_unzip(url, dest_folder='.'):
    """
    Download a zip file from a URL and unzip it in the destination folder with progress indication.
    """
    # Send a GET request to the URL
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check if the download was successful

        print("Downloading from OSF...")
        with open("downloaded_file.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=4096):
                file.write(chunk)
        
        print("Download completed. Extracting files...")
        with zipfile.ZipFile("downloaded_file.zip") as thezip:
            thezip.extractall(path=dest_folder)
            print("Extraction completed.")

        print("Cleaning up...")
        os.remove("downloaded_file.zip")

        print("Done")

if __name__ == "__main__":
    # Download and unzip the file to the current directory
    download_and_unzip(zip_url)
