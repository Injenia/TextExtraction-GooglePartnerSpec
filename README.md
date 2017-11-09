# Prerequisiti
python 2.7
google cloud sdk installato e configurato con infocamere-poc come progetto corrente.
opzionale per OCR: imagemagick (linux)

# Installazione dipendenze python
pip install -r requirements.txt
per demo: pip install jupyter

# Demo
Per la demo, aprire in un notebook jupyter il file demo.ipynb

# Backend
Per lanciare il servizio FLASK:

export FLASK_APP=backend.py
flask run --host=0.0.0.0