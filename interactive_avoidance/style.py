# Code here is borrowed and annotated from Opinionated (https://github.com/MNoichl/opinionated)
import io
import os
import zipfile

import matplotlib.pyplot as plt
import requests


def download_googlefont(font: str = "Roboto Condensed") -> None:
    """Download a font from Google Fonts and save it in the fonts folder.

    Args:
        font (str, optional): The name of the font to download from Google Fonts. Defaults to 'Roboto Condensed'.
    """
    url = f'https://fonts.google.com/download?family={font.replace(" ", "%20")}'
    r = requests.get(url)

    if r.status_code != 200:
        print(f"Failed to download {font} from Google Fonts.")
        return

    z = zipfile.ZipFile(io.BytesIO(r.content))
    font_folder = "./fonts"

    if not os.path.exists(font_folder):
        os.makedirs(font_folder)

    z.extractall(font_folder)
    print(f"Font saved to: {font_folder}")


def set_style(style_path: str = "../style.mplstyle", font: str = "Heebo") -> None:
    """Set the Matplotlib style and download the specified font from Google Fonts.

    Args:
        style_path (str, optional): The path to the Matplotlib style file. Defaults to '../style.mplstyle'.
        font (str, optional): The name of the font to download from Google Fonts. Defaults to 'Heebo'.
    """
    download_googlefont(font)

    # Read the original style file and replace the font.family line with the new font
    with open(style_path, "r") as f:
        style_lines = f.readlines()

    new_style_lines = [
        line.replace("font.family: sans-serif", f"font.family: {font}")
        if line.startswith("font.family")
        else line
        for line in style_lines
    ]

    # Use a temporary style file with updated font family
    with open("temp_style.mplstyle", "w") as f:
        f.writelines(new_style_lines)

    plt.style.use("temp_style.mplstyle")
    print(f"Matplotlib style set to: {style_path} with font {font}")
