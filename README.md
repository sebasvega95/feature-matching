# Feature matching using SURF descriptors and geometric properties

## Get the images
Testing is done using the [Stanford Mobile Visual Search Dataset](http://web.cs.wpi.edu/~claypool/mmsys-dataset/2011/stanford/),
specifically the [book covers](http://web.cs.wpi.edu/~claypool/mmsys-dataset/2011/stanford/mvs_images/book_covers.tgz) section.

## Setup

Create the virtual environment

```bash
  sudo apt install python-virtualenv
  virtualenv -p /usr/bin/python3 env
```

Install dependencies
```bash
  env/bin/pip install -r requirements.txt
```

## Run

```bash
  chmod +x main.py
  ./main.py
```
