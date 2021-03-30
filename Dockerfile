FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install --upgrade cython
RUN pip install hdbscan --no-cache-dir --no-binary :all: --no-build-isolation
EXPOSE 80
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]