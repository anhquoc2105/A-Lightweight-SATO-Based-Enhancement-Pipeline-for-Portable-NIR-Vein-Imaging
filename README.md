# Vein Image Processing Setup Guide (Windows & macOS)

This project processes a hand image and displays each processing step for vein
segmentation. The program uses OpenCV, NumPy, Matplotlib, and scikit-image.

Main processing steps:

- Resize the input image
- Segment the hand area with GrabCut
- Convert the segmented hand to grayscale
- Enhance contrast with CLAHE
- Detect vein-like structures with the Sato filter
- Clean the vein mask with morphology operations
- Display the final vein overlay on the hand image

## Requirements

- Python 3.8 or higher
- pip
- A terminal opened in this project folder:

```bash
C:\Users\Dell\Desktop\Vein
```

Check your Python version:

**Windows**

```powershell
python --version
```

**macOS**

```bash
python3 --version
```

## 1. Create a virtual environment

Open a terminal in the project folder and run:

**Windows**

```powershell
python -m venv .venv
```

**macOS**

```bash
python3 -m venv .venv
```

This creates a folder named `.venv` containing an isolated Python environment
for this project.

## 2. Activate the virtual environment

### Windows - Command Prompt

```bat
.venv\Scripts\activate
```

### Windows - PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If you get an execution policy error in PowerShell, run this first, then try
activating again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS - Terminal

```bash
source .venv/bin/activate
```

Once activated, you should see `(.venv)` at the beginning of your terminal
prompt.

## 3. Install packages from requirements.txt

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

Installed packages:

- `opencv-python`
- `matplotlib`
- `numpy`
- `scikit-image`

## 4. Verify the installation

Quickly test the required imports:

```bash
python -c "import cv2, matplotlib, numpy, skimage; print('Setup successful!')"
```

On macOS, use this if `python` is not recognized:

```bash
python3 -c "import cv2, matplotlib, numpy, skimage; print('Setup successful!')"
```

## 5. Run the program

Make sure the virtual environment is active before running the program.

Run with the default image:

```bash
python main.py
```

The default input image is:

```bash
image/1.png
```

Run with a custom input image:

**Windows**

```powershell
python main.py --file .\image\1.png
```

**macOS**

```bash
python main.py --file ./image/1.png
```

You can replace `image/1.png` with any image path you want to process.

## CLI

```bash
python main.py --file path/to/input.png
```

Arguments:

- `--file`, `-f`: path to the input image. If omitted, the program uses
  `image/1.png`.

Examples:

```bash
python main.py -f ./image/10.png
python main.py --file ./image/37.png
```

## Project Structure

```text
Vein/
├── image/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── main.py
├── README.md
└── requirements.txt
```

## 6. Deactivate the virtual environment

When you are done working, deactivate the virtual environment with:

```bash
deactivate
```

## Notes

- Every time you open a new terminal, reactivate the virtual environment before
  running code or installing packages.
- Do not commit the `.venv/` folder to Git.
- If you add new packages later, update `requirements.txt` with:

```bash
pip freeze > requirements.txt
```
