import gemmi
import os
import requests
import zipfile
from datetime import datetime
from flask import jsonify, render_template, flash, request, send_from_directory, redirect, url_for, Response, Flask, Markup
from multiprocessing import Process, Manager
from random import random
from glob import glob
from time import time
from ppropt import PRO


application = Flask(__name__)
application.jinja_env.trim_blocks = True
application.jinja_env.lstrip_blocks = True
application.config['SECRET_KEY'] = str(random())
root_dir = os.path.dirname(os.path.abspath(__file__))

queue = Manager().list()
running = Manager().list()
optimisers = []
number_of_processes = 1
number_of_cpu = 32


def create_mmcif(original_CIF_file, optimised_PDB_file, optimised_CIF_file, code):
    structure = gemmi.read_pdb(optimised_PDB_file)
    structure.setup_entities()
    structure.assign_label_seq_id()
    block = structure.make_mmcif_block()
    block.find_mmcif_category('_chem_comp.').erase() # remove pesky _chem_comp category >:(
    response = requests.get(f'https://alphafold.ebi.ac.uk/files/AF-{code}-F1-model_v4.cif')
    with open(original_CIF_file, 'w') as cif_file:
        cif_file.write(response.text)
    document = gemmi.cif.read_string(response.text)
    sole_block = document.sole_block()
    ma_qa_metric_prefix = '_ma_qa_metric'
    ma_qa_metric_local_prefix = '_ma_qa_metric_local'
    ma_qa_metric_global_prefix = '_ma_qa_metric_global'
    categories = {
        ma_qa_metric_prefix: sole_block.get_mmcif_category(ma_qa_metric_prefix),
        ma_qa_metric_local_prefix: sole_block.get_mmcif_category(ma_qa_metric_local_prefix),
        ma_qa_metric_global_prefix: sole_block.get_mmcif_category(ma_qa_metric_global_prefix)
    }
    asym_id = block.get_mmcif_category('_struct_asym').get('id')[0]
    length = len(categories[ma_qa_metric_local_prefix]['label_asym_id'])
    categories[ma_qa_metric_local_prefix]['label_asym_id'] = [asym_id] * length
    for name, data in categories.items():
        block.set_mmcif_category(name, data)
    block.write_file(optimised_CIF_file)


def optimise_structures():
    while len(queue):
        ID = queue.pop(0)
        running.append(ID)
        code, ph = ID.split('_')
        data_dir = f'{root_dir}/calculated_structures/{ID}'
        pdb_file = f'{data_dir}/{code}.pdb'
        pdb_file_with_hydrogens = f'{data_dir}/{code}_added_H.pdb'

        # protonate structure
        os.system(f'/opt/venv/bin/pdb2pqr30 --log-level DEBUG --titration-state-method propka '
                  f'--with-ph {ph} --pdb-output {pdb_file_with_hydrogens} {pdb_file} '
                  f'{data_dir}/{code}.pqr > {data_dir}/propka.log 2>&1 ')

        # optimise structure
        PRO(f"{data_dir}/optimisation", pdb_file_with_hydrogens, number_of_cpu).optimise()

        # create mmcif
        optimised_PDB_file = f'{data_dir}/optimisation/optimised_PDB/{code}_added_H_optimised.pdb'
        optimised_CIF_dir = f'{data_dir}/optimisation/optimised_CIF'
        optimised_CIF_file = f'{optimised_CIF_dir}/{code}_added_H_optimised.cif'
        original_CIF_file = f'{data_dir}/{code}.cif'
        os.mkdir(optimised_CIF_dir)
        create_mmcif(original_CIF_file, optimised_PDB_file, optimised_CIF_file, code)
        running.remove(ID)


def job_status(ID: str):
    if os.path.isfile(f'{root_dir}/calculated_structures/{ID}/optimisation/optimised_PDB/{ID.split("_")[0]}_added_H_optimised.pdb'):
        return "finished"
    elif os.path.isdir(f'{root_dir}/calculated_structures/{ID}'):
        if ID in queue:
            return "queued"
        else:
            return "running"
    return "unsubmitted"

@application.route('/', methods=['GET', 'POST'])
def main_site():

    if request.method == 'POST':

        # load user input
        code = request.form['code'].strip().upper()  # UniProt code, not case-sensitive
        code = code.replace("AF-","").replace("-F1", "")  # Also AlphaFold DB identifiers are supproted (e.g. AF-A8H2R3-F1)
        ph = request.form['ph']
        if "." not in ph:
            ph = ph + ".0"
        ID = f'{code}_{ph}'

        # log access
        with open(f'{root_dir}/calculated_structures/logs.txt', 'a') as log_file:
            log_file.write(f'{request.remote_addr} {code} {ph} {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')

        status = job_status(ID)

        if status == "finished":
            return redirect(url_for('results', ID=ID))

        elif status in ["queued", "running"]:
            flash(Markup(f'Optimisation of structure <strong>{code}</strong> with pH <strong>{ph}</strong> is already submitted. '
                         f'For job status visit <a href="https://fffold.biodata.ceitec.cz/results?ID={ID}" class="alert-link"'
                         f'target="_blank" rel="noreferrer">https://fffold.biodata.ceitec.cz/results?ID={ID}</a>.'), 'info')
            return render_template('index.html', running=len(running), queued=len(queue))

        elif status == "unsubmitted":

            # download pdb
            response = requests.get(f'https://alphafold.ebi.ac.uk/files/AF-{code}-F1-model_v4.pdb')
            if response.status_code != 200:
                flash(Markup(f'The structure with code <strong>{code}</strong> '
                             f'is either not found in AlphaFold DB or the code is entered in the wrong format. '
                             f'UniProt code is allowed only in its short form (e.g. A0A1P8BEE7, B7ZW16). '
                             f'Other notations (e.g. A0A159JYF7_9DIPT, Q8WZ42-F2) are not supported. '
                             f'An alternative option is AlpfaFold DB Identifier (e.g. AF-L8BU87-F1).'), 'warning')
                return render_template('index.html', running=len(running), queued=len(queue))
            data_dir = f'{root_dir}/calculated_structures/{ID}'
            os.mkdir(data_dir)
            with open(f'{data_dir}/{code}.pdb', 'w') as pdb:
                pdb.write(response.text)

            # create and submit job
            global optimisers
            optimisers = [optimiser for optimiser in optimisers if optimiser.is_alive()]
            queue.append(ID)
            if len(optimisers) < number_of_processes:
                optimiser = Process(target=optimise_structures)
                optimiser.start()
                optimisers.append(optimiser)
            return redirect(url_for('results', ID=ID))

    return render_template('index.html', running=len(running), queued=len(queue))



@application.route('/results')
def results():
    ID = request.args.get('ID')

    try:
        code, ph = ID.split('_')
    except:
        flash(Markup('The ID was entered in the wrong format. '
                     'The ID should be of the form <strong>&ltUniProt code&gt_&ltph&gt.'), 'danger')
        return redirect(url_for('main_site'))

    status = job_status(ID)

    if status == "unsubmitted":
        flash(Markup(f'There are no results for structure with UniProt <strong>{code}</strong> and pH <strong>{ph}</strong>.'), 'danger')
        return redirect(url_for('main_site'))

    if status == "queued":
        status = f"Optimisation is queued."
        return render_template('queued.html',
                               code=code,
                               ph=ph)

    elif status == "running":
        n_optimised_residues = len(glob(f'{root_dir}/calculated_structures/{ID}/optimisation/sub_*/xtbopt.pdb'))
        total_n_residues = int(open(f'{root_dir}/calculated_structures/{ID}/{code}.pdb', "r").readlines()[-4][22:26])
        percent_value = round(n_optimised_residues/(total_n_residues*0.01))
        percent_text = f"{n_optimised_residues}/{total_n_residues}"
        return render_template('running.html',
                               ID=ID,
                               code=code,
                               ph=ph,
                               percent_value = percent_value,
                               percent_text = percent_text)

    return render_template('results.html',
                           ID=ID,
                           code=code,
                           ph=ph)


@application.route('/api/running_progress', methods=['GET'])
def running_progress():
    ID = request.args.get('ID')

    try:
        code, _ = ID.split('_')
    except:
        return Response('The ID was entered in the wrong format. '
                     'The ID should be of the form <strong>&ltUniProt code&gt_&ltph&gt.',
                     status=404,
                     mimetype='text/plain')
    
    status = job_status(ID)
    response = { 'status': status }
    
    if status == 'unsubmitted':
        return jsonify(response)
    if status == 'queued':
        return jsonify(response)

    if status == 'finished':
        response.update({
            'url': url_for('results', ID=ID)
        })
        return jsonify(response)
        
    n_optimised_residues = len(glob(f'{root_dir}/calculated_structures/{ID}/optimisation/sub_*/xtbopt.pdb'))
    total_n_residues = int(open(f'{root_dir}/calculated_structures/{ID}/{code}.pdb', "r").readlines()[-4][22:26])
    percent_value = f"{round(n_optimised_residues/(total_n_residues*0.01))}"
    percent_text = f"{n_optimised_residues}/{total_n_residues}"
        
    response.update({
        'percent_value': percent_value,
        'percent_text': percent_text
    })
    
    return jsonify(response)
    

@application.route('/download_files')
def download_files():
    ID = request.args.get('ID')
    code, _ = ID.split("_")
    data_dir = f'{root_dir}/calculated_structures/{ID}'
    with zipfile.ZipFile(f'{data_dir}/{ID}.zip', 'w') as zip:
        zip.write(f'{data_dir}/optimisation/optimised_PDB/{code}_added_H_optimised.pdb',f'{code}_optimised.pdb')
        zip.write(f'{data_dir}/optimisation/optimised_CIF/{code}_added_H_optimised.cif', f'{code}_optimised.cif')
        zip.write(f'{data_dir}/{code}.pdb', f'{code}_original.pdb')
        zip.write(f'{data_dir}/{code}.cif', f'{code}_original.cif')
    return send_from_directory(data_dir, f'{ID}.zip', as_attachment=True)


@application.route('/optimised_structure/<ID>')
def get_optimised_structure(ID: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimisation/optimised_CIF/{ID.split("_")[0]}_added_H_optimised.cif'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')


@application.route('/original_structure/<ID>')
def get_original_structure(ID: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimisation/inputed_PDB/{ID.split("_")[0]}_added_H.pdb'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')


@application.route('/residues_logs/<ID>')
def get_residues_logs(ID: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimisation/residues.logs'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')


@application.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
