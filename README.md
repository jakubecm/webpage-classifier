# Repository Overview

This repository contains a machine learning-based classifier integrated with a web backend and a Chrome extension, created as a part of my bachelors theses on webpage topic detection via machine learning methods.

It is able to detect the following categories:

- Adult
- Computers
- Games
- Health
- News
- Recreation
- Reference
- Science
- Shopping

The classifier used is an SVM, and it is about 80% accurate (weighted precision).

 The project is divided into two main parts:  
1. **Backend**: A Flask-based server that handles the classification logic.  
2. **Chrome Extension**: A user interface for interacting with the classifier directly from the browser.

It also contains a folder called **Notebooks**, where you can find the training or experimenting notebooks.

## Getting Started

Follow the steps below to set up and use the project.

### Prerequisites

Ensure you have the following installed on your system:  
- Python 3.8+  (ideally 3.10.15 as it is tested on this version)
- pip (Python package manager)  
- Google Chrome browser  

### Installation

1. Clone the repository:  
    ```bash
    git clone https://github.com/jakubecm/webpage-classifier.git
    cd your-repo
    ```

2. Install the required Python libraries (I recommend using a venv if possible):  
    ```bash
    pip install -r requirements.txt
    ```

### Running the Backend

1. Navigate to the backend directory:  
    ```bash
    cd BrowserExtension
    ```

2. Start the Flask server locally:  
    ```bash
    flask run
    ```

    The server will start running at `http://127.0.0.1:5000`.

### Setting Up the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.  
2. Enable **Developer mode** (toggle in the top-right corner).  
3. Click **Load unpacked** and select the `chrome_extension_part` folder from this repository.  
4. The extension will now be loaded and ready to use.

### Using the Classifier

1. Ensure the Flask backend is running.  
2. Open the Chrome browser and use the extension to interact with the classifier.  
3. The extension will send requests to the backend and display the classification results.

## Contributing

In case you are interesting in the project itself, feel free to clone/fork and use it or even improve it.
The dataset used for training is of course too large to be uploaded here, in the following weeks I aim to upload it to Zenodo or Kaggle, so far it is only exsistant as a .sql file. 

## License
This project is licensed under the [GNU General Public License v3.0](LICENSE).
