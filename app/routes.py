import os
import zipfile
from random import random
from time import time

from flask import render_template, flash, request, send_from_directory, redirect, url_for, Response, Flask, Markup
import requests
import gemmi

from src.calculation import Calculation
from src.input_validators import valid_pH,  valid_alphafold_request

application = Flask(__name__)
application.jinja_env.trim_blocks = True
application.jinja_env.lstrip_blocks = True
application.config['SECRET_KEY'] = str(random())
root_dir = os.path.dirname(os.path.abspath(__file__))


def already_calculated(ID):
    path = f'{root_dir}/calculated_structures/{ID}'
    if os.path.isdir(path):
        if os.path.isfile(f'{path}/optimization/optimized_PDB/{ID.split("_")[0]}_added_H_optimized.pdb'):
            return True
        elif time() - os.stat(path).st_mtime > 10000:
            # for case that results directory exists without optimized structures
            # it means that something unexpected happen during calculation
            os.system(f'rm -r {root_dir}/calculated_structures/{ID}')
    return False


def is_running(ID):
    path = f'{root_dir}/calculated_structures/{ID}'
    if os.path.isdir(path):
        if os.path.isfile(f'{path}/optimization/optimized_PDB/{ID.split("_")[0]}_added_H_optimized.pdb'):
            return False
        elif time() - os.stat(path).st_mtime > 10000:
            return False
        return True
    return False

@application.route('/', methods=['GET', 'POST'])
def main_site():
    if request.method == 'POST':
        code = request.form['code'].strip().upper() # UniProt code, not case-sensitive
        code = code.replace("AF-","").replace("-F1", "") # Also AlphaFold DB identifiers are supproted (e.g. AF-A8H2R3-F1)

        if request.form['action'] == 'settings':
            return render_template('settings.html',
                                   code=code)

        elif request.form['action'] == 'optimize structure':
            ph = request.form['ph']

            ph, is_ph_valid = valid_pH(ph)
            if not is_ph_valid:
                message = 'pH must be a float value from 0 to 14!'
                flash(message, 'warning')
                return render_template('index.html',
                                       code=code)

            ID = f'{code}_{ph}'

            # check whether the structure is currently calculated
            if is_running(ID):
                message = Markup(f'The structure is just optimized. '
                             f'For results visit  <a href="https://fffold.ncbr.muni.cz/results?ID={ID}" class="alert-link"'
                             f'target="_blank" rel="noreferrer">https://fffold.ncbr.muni.cz/results?ID={ID}</a>'
                             f' after a while.')
                flash(message, 'info')
                return render_template('index.html')

            if already_calculated(ID):
                return redirect(url_for('results',
                                        ID=ID))

            if not valid_alphafold_request(code):
                message = Markup(f'The structure with code <strong>{code}</strong> in prediction version <strong>4</strong> '
                      f'is either not found in AlphaFoldDB or the code is entered in the wrong format. '
                      f'UniProt code is allowed only in its short form (e.g. A0A1P8BEE7, B7ZW16). '
                      f'Other notations (e.g. A0A159JYF7_9DIPT, Q8WZ42-F2) are not supported. '
                      f'An alternative option is AlpfaFold DB Identifier (e.g. AF-L8BU87-F1).')
                flash(message, 'warning')
                return render_template('index.html')

            # start calculation
            return render_template('computation_progress.html',
                                   ID=ID,
                                   code=code,
                                   ph=ph)
    return render_template('index.html')


@application.route('/calculation', methods=['POST'])
def calculation():

    calculation = Calculation(request.args.get('ID'),
                              request.remote_addr,
                              "empirical_method",
                              root_dir)
    calculation.download_PDB()
    calculation.protonate_structure()
    calculation.optimize_structure()
    return redirect(url_for('results', ID=calculation.ID))


@application.route('/progress')
def progress():
    return open(f'{root_dir}/calculated_structures/{request.args.get("ID")}/page_log.txt', 'r').read()


@application.route('/results')
def results():
    ID = request.args.get('ID')

    try:
        code, ph = ID.split('_')
    except:
        message = Markup('The ID was entered in the wrong format. The ID should be of the form <strong>&ltUniProt code&gt_&ltph&gt.')
        flash(message, 'danger')
        return redirect(url_for('main_site'))


    if not already_calculated(ID) and not is_running(ID):
        message = Markup(f'There are no results for structure with UniProt <strong>{code}</strong> and pH <strong>{ph}</strong>.')
        flash(message, 'danger')
        return redirect(url_for('main_site'))


    return render_template('results.html',
                           ID=ID,
                               code=code,
                               ph=ph)



@application.route('/download_files')
def download_files():
    ID = request.args.get('ID')
    code, _, _ = ID.split('_')
    data_dir = f'{root_dir}/calculated_structures/{ID}'
    with zipfile.ZipFile(f'{data_dir}/{ID}.zip', 'w') as zip:
        zip.write(f'{data_dir}/optimization/optimized_PDB/{ID.split("_")[0]}_added_H_optimized.pdb',f'{ID.split("_")[0]}_added_H_optimized.pdb')
        zip.write(f'{data_dir}/optimization/optimized_CIF/{ID.split("_")[0]}_added_H_optimized.cif', f'{ID.split("_")[0]}_added_H_optimized.cif')
    return send_from_directory(data_dir, f'{ID}.zip', as_attachment=True)

def add_AF_confidence_score(write_block, ID):
    response = requests.get(f'https://alphafold.ebi.ac.uk/files/AF-{ID}-F1-model_v4.cif')
    document = gemmi.cif.read_string(response.text)
    block = document.sole_block()

    ma_qa_metric_prefix = '_ma_qa_metric'
    ma_qa_metric_local_prefix = '_ma_qa_metric_local'
    ma_qa_metric_global_prefix = '_ma_qa_metric_global'

    categories = {
        ma_qa_metric_prefix: block.get_mmcif_category(ma_qa_metric_prefix),
        ma_qa_metric_local_prefix: block.get_mmcif_category(ma_qa_metric_local_prefix),
        ma_qa_metric_global_prefix: block.get_mmcif_category(ma_qa_metric_global_prefix)
    }

    asym_id = write_block.get_mmcif_category('_struct_asym').get('id')[0]

    length = len(categories[ma_qa_metric_local_prefix]['label_asym_id'])
    categories[ma_qa_metric_local_prefix]['label_asym_id'] = [asym_id] * length

    for name, data in categories.items():
        write_block.set_mmcif_category(name, data)
    

def create_mmcif(input_file, output_file, ID):
    structure = gemmi.read_pdb(input_file)
    structure.setup_entities()
    structure.assign_label_seq_id()
    block = structure.make_mmcif_block()
    block.find_mmcif_category('_chem_comp.').erase() # remove pesky _chem_comp category >:(
    add_AF_confidence_score(block, ID)
    block.write_file(output_file)

@application.route('/structure/<ID>/<FORMAT>')
def get_structure(ID: str,
                  FORMAT: str):
    input_file = f'{root_dir}/calculated_structures/{ID}/optimization/optimized_PDB/{ID.split("_")[0]}_added_H_optimized.pdb'
    if FORMAT == 'pdb':
        return Response(open(input_file, 'r').read(), mimetype='text/plain')
    elif FORMAT == 'cif':
        output_dir = f'{root_dir}/calculated_structures/{ID}/optimization/optimized_CIF'
        output_file = f'{output_dir}/{ID.split("_")[0]}_added_H_optimized.cif'
        try:
            os.mkdir(output_dir)
        except:
            pass
        create_mmcif(input_file, output_file, ID.split("_")[0])
        return Response(open(output_file, 'r').read(), mimetype='text/plain')

    return Response('', mimetype='text/plain')

@application.route('/original_structure/<ID>/<FORMAT>')
def get_original_structure(ID: str,
                           FORMAT: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimization/inputed_PDB/{ID.split("_")[0]}_added_H.pdb'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')

@application.route('/residues_logs/<ID>')
def get_residues_logs(ID: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimization/residues.logs'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')

@application.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
