# INNO 2021 Backend Repository

This repository can either be installed and used as a Python module or as a decision mining API.

## Installing as a module

```shell
git clone https://github.com/HU-DIPO/DecisionMiningFuzzy.git
cd DecisionMiningFuzzy\\back-end
pip install .
```

Or as a developer:

```shell
git clone https://github.com/HU-DIPO/DecisionMiningFuzzy.git
cd DecisionMiningFuzzy\\back-end
pip install .[dev]
```

After this, simply `import decision_mining` in your own Python code.

## Running API individually

As the name implies, the backend can also be used together with the frontend. It can also be run individually, either locally or through a docker container.

### Running with Docker

```shell
git clone https://github.com/HU-DIPO/DecisionMiningFuzzy.git
cd DecisionMiningFuzzy\\back-end
docker build -t DMFuzzy/back:v1 .
docker run -p 5000:5000 DMFuzzy/back:v1
```

And then go to `localhost:5000`.

### Running with Flask

Repeat instructions from `Installing as a module`, then:

```shell
flask run
```

And then go to `localhost:5000`.

## Generate the documentation

To generate the documentation of this project we use Sphinx.

After installing decision_mining as dev, do:

```shell
DecisionMiningFuzzy\\back-end\\srcdocs\\make.bat html
```
