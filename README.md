# Python Virtual Environment Setup

This project uses a Python virtual environment to manage dependencies.

## Setup Instructions

### 1. Create Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Deactivate Virtual Environment
When you're done working:
```bash
deactivate
```

### 5. Dataset download
Download the dataset from: https://www.kaggle.com/datasets/mrdheer/cars-dataset

## Notes
- The virtual environment folder `venv/` is typically added to `.gitignore`
- Always activate the virtual environment before running Python scripts
- Use `pip freeze > requirements.txt` to save current dependencies