# CS 5890 - Final Project

## Setup

This project is built in Python. 

### Clone the Repo

To use the software, you'll need to clone this repo to your computer. 

```bash
git clone https://github.com/masoncfrancis/cs5890-iep504planorganizer.git
```

### Install Dependencies

This project has several dependencies. You can install them by running this command:

```bash
python3 -m pip install -r requirements.txt
```

### Configure Your OpenAI API Key

You will need an OpenAI account and API key to run this software. You can get an API key [here](https://platform.openai.com/api-keys).

Save your API key in `secrets.json`. When finished, the file contents should look like this:

```json
{
    "openaiKey": "YOUR-KEY-HERE"
}
```

**Note: Please be aware that the use of this software may incur charges through your it's use of the OpenAI API.**

### Run The Server

Run the main server by running this command:

```bash
python3 app.py
```

The script should output a localhost URL where you can use the software. 


## Credit

I based my project initially off of code available [here](https://github.com/sunilkumardash9/Pdf-GPT) in 
order to have a web interface and be able to ingest PDFs into the LLM. Information about the licensing
terms used is available in [opensource.md](opensource.md).

I also used ChatGPT in making changes to and understanding the GUI in Gradio. I used OpenAI's API documentation
to integrate GPT-4 into this project. 

## Licensing

This software is not available for commercial or non-commercial use by any individual or organization. 
