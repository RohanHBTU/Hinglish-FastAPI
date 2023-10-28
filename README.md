# Hinglish-Transformer
It uses a Encoder-Decoder architecture with Bahdanau attention mechanism using GRU RNN cell. This has been tested and resulted with a mean BLEU score of 50%. It has been Deployed using FastAPI Framework.

To setup this repo into your local machine:
1. Fork the repository
2. Open git bash and clone it using ```git clone https://github.com/your_username/Hinglish-FastAPI.git```
3. Create a virtual environment using ```py -m venv env``` in Command Prompt (Note: Make sure you have pip installed virtualenv package)
4. Activate the environment using ```.\env\Scripts\activate```
3. Pip install all the python packages mentioned in the requirements.txt using ```py -m pip install -r requirements.txt```
4. With Command Prompt opened in present working directory
5. Execute:
```uvicorn app:app --reload```

Your app will start at a localhost port(only for development).
