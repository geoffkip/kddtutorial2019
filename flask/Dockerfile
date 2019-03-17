# Python base image
FROM python:3.6

# Updating repository sources
RUN apt-get update

# Copying requirements.txt file
COPY requirements.txt requirements.txt

# pip install
RUN pip install --no-cache -r requirements.txt

# Copy necessary app items
COPY app.py app.py
COPY predict.py predict.py
COPY kidney_disease.csv kidney_disease.csv

# Exposing ports
EXPOSE 5000

# Start app
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
