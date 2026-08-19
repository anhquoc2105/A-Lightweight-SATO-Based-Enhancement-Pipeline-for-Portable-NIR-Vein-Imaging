# A Lightweight SATO-Based Enhancement Pipeline for Portable NIR Vein Imaging.

This guide walks you through creating a Python virtual environment, installing the required packages, and running the program on both Windows and macOS.

## Requirements

- Python 3.8 or higher installed on your machine
- Check your Python version:

**Windows**
```bash
python --version
```

**macOS**
```bash
python3 --version
```

## 1. Create a virtual environment

Open a terminal in your project folder and run:

**Windows**
```bash
python -m venv .venv
```

**macOS**
```bash
python3 -m venv .venv
```

This creates a folder named `venv` containing an isolated Python environment for your project.

## 2. Activate the virtual environment

### Windows
```bash
.venv\Scripts\activate
```

### macOS

```bash
source .venv/bin/activate
```

Once activated, you'll see `(venv)` appear at the beginning of your command line prompt.

## 3. Install packages from requirements.txt

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```


## 4. Run the program

With the virtual environment activated and all packages installed:

**Run with the default image:**

```bash
python main.py
```

**Run with a custom input image:**

```bash
python main.py --file ./image/1.png
```

### CLI

```bash
python main.py --file path/to/input.png
```

Arguments:

- `--file`, `-f`: path to the input image. If omitted, the program uses `image/1.png`.

## 5. Deactivate the virtual environment

When you're done working, deactivate with:

```bash
deactivate
```

