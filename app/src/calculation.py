import os
from datetime import datetime
from time import time

import requests

from .logs import Logs

from .pro.pro import PRO


class Calculation:
    def __init__(self,
                 ID: str,
                 remote_addr: str,
                 empirical_method: str,
                 root_dir: str):
        self.ID = ID
        self.empirical_method = empirical_method
        self.root_dir = root_dir
        self.code, self.ph = self.ID.split('_')
        self.data_dir = f'{self.root_dir}/calculated_structures/{self.ID}'
        self.pdb_file = f'{self.data_dir}/{self.code}.pdb' # original pdb from alphafold, without hydrogens
        self.mmcif_file = f'{self.data_dir}/{self.code}.cif' # original mmCIF from alphafold, without hydrogens
        self.pdb_file_with_hydrogens = f'{self.data_dir}/{self.code}_added_H.pdb'
        self.pqr_file = f'{self.data_dir}/{self.code}.pqr'
        self.logs = Logs(data_dir=self.data_dir, empirical_method=self.empirical_method)
        os.mkdir(self.data_dir)
        os.mknod(f'{self.data_dir}/page_log.txt')
        with open(f'{self.root_dir}/calculated_structures/logs.txt', 'a') as log_file:
            log_file.write(f'{remote_addr} {self.code} {self.ph} {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')

    def download_PDB(self):
        self.logs.add_log('Structure download...')
        s = time()
        response = requests.get(f'https://alphafold.ebi.ac.uk/files/AF-{self.code}-F1-model_v4.pdb')
        with open(f'{self.pdb_file}', 'w') as pdb_file:
            pdb_file.write(response.text)
        response = requests.get(f'https://alphafold.ebi.ac.uk/files/AF-{self.code}-F1-model_v4.cif')
        with open(f'{self.mmcif_file}', 'w') as mmcif_file:
            mmcif_file.write(response.text)
        self.logs.add_log(f'Structure downloaded. ({round(time() - s, 2)}s)')

    def protonate_structure(self):
        self.logs.add_log('Protonation of structure...')
        s = time()
        os.system(f'/opt/venv/bin/pdb2pqr30 --log-level DEBUG --noopt --titration-state-method propka '
                  f'--with-ph {self.ph} --pdb-output {self.pdb_file_with_hydrogens} {self.pdb_file} '
                  f'{self.pqr_file} > {self.data_dir}/propka.log 2>&1 ')
        self.logs.add_log(f'Structure protonated. ({round(time() - s, 2)}s)')




    def optimize_structure(self):
        self.logs.add_log('Optimization of structure...')
        s = time()
        PRO(f"{self.data_dir}/optimization",
            self.pdb_file_with_hydrogens).optimize()
        self.logs.add_log(f'Structure optimized. ({round(time() - s, 2)}s)')

