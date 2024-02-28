#!/bin/bash

# update system and install deps
sudo apt update -y && sudo apt upgrade -y
sudo apt -y install python3-pip python3-venv apache2 libapache2-mod-wsgi-py3 git

# clone repo
cd /opt
sudo git clone --recurse-submodules --depth 1 https://github.com/sb-ncbr/FFFold

# install python deps
sudo python3 -m venv venv
source venv/bin/activate
sudo chown -R ubuntu:ubuntu /opt
pip install -r FFFold/requirements.txt

# install xtb and openbabel
sudo mkdir -p miniconda3
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
sudo bash miniconda3/miniconda.sh -b -u -p miniconda3
sudo rm miniconda3/miniconda.sh
sudo miniconda3/bin/conda install -y -c conda-forge xtb
sudo miniconda3/bin/conda install -y openbabel=3.1.1

# set paths to xtb in ppropt.py
sudo sed -i -e 's/xtb substructure/\/opt\/miniconda3\/bin\/xtb substructure/g' /opt/FFFold/app/ppropt/ppropt.py

# setup web server
sudo rm -f /etc/apache2/sites-available/*
sudo cp FFFold/FFFold.conf /etc/apache2/sites-available/
sudo chown -R www-data:www-data /opt
sudo chmod o+rx FFFold/app/FFFold.wsgi
sudo chmod o+rx FFFold/app/routes.py
sudo a2ensite FFFold.conf
sudo a2enmod ssl
sudo a2enmod brotli
sudo a2enmod http2
sudo systemctl restart apache2

