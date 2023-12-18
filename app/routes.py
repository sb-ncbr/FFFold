import gemmi
import os
import requests
import zipfile
from datetime import datetime
from flask import render_template, flash, request, send_from_directory, redirect, url_for, Response, Flask, Markup
from multiprocessing import Process, Manager
from random import random
from time import time

from ppropt.ppropt import PRO

application = Flask(__name__)
application.jinja_env.trim_blocks = True
application.jinja_env.lstrip_blocks = True
application.config['SECRET_KEY'] = str(random())
root_dir = os.path.dirname(os.path.abspath(__file__))

queue = Manager().list()
calculation_times = Manager().dict()
running = Manager().dict()
optimizers = []
number_of_processes = 8


def create_mmcif(input_file, output_file, code):
    def _add_AF_confidence_score(write_block, code: str):
        response = requests.get(f'https://alphafold.ebi.ac.uk/files/AF-{code}-F1-model_v4.cif')
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
    structure = gemmi.read_pdb(input_file)
    structure.setup_entities()
    structure.assign_label_seq_id()
    block = structure.make_mmcif_block()
    block.find_mmcif_category('_chem_comp.').erase() # remove pesky _chem_comp category >:(
    _add_AF_confidence_score(block, code)
    block.write_file(output_file)


def optimize_structures():
    while len(queue):
        ID = queue.pop(0)
        running[ID] = time() + calculation_times[ID]
        code, ph = ID.split('_')
        data_dir = f'{root_dir}/calculated_structures/{ID}'
        pdb_file = f'{data_dir}/{code}.pdb'
        pdb_file_with_hydrogens = f'{data_dir}/{code}_added_H.pdb'

        # protonate structure
        os.system(f'/opt/venv/bin/pdb2pqr30 --log-level DEBUG --noopt --titration-state-method propka '
                  f'--with-ph {ph} --pdb-output {pdb_file_with_hydrogens} {pdb_file} '
                  f'{data_dir}/{code}.pqr > {data_dir}/propka.log 2>&1 ')

        # optimize structure
        PRO(f"{data_dir}/optimization",
            pdb_file_with_hydrogens).optimize()

        # create mmcif
        input_file = f'{data_dir}/optimization/optimized_PDB/{code}_added_H_optimized.pdb'
        output_dir = f'{data_dir}/optimization/optimized_CIF'
        output_file = f'{output_dir}/{code}_added_H_optimized.cif'
        os.mkdir(output_dir)
        create_mmcif(input_file, output_file, code)
        del running[ID]
        del calculation_times[ID]


def job_status(ID: str):
    if os.path.isfile(f'{root_dir}/calculated_structures/{ID}/optimization/optimized_PDB/{ID.split("_")[0]}_added_H_optimized.pdb'):
        return "finished"
    elif os.path.isdir(f'{root_dir}/calculated_structures/{ID}'):
        if ID in queue:
            return "queued"
        else:
            return "running"
    return "unsubmitted"

def server_status():
    if len(running) == 0:
        return "FREE"
    elif len(running) < number_of_processes:
        return "PARTIALLY FREE"
    else:
        return "BUSY"

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
            flash(Markup(f'Optimization of structure <strong>{code}</strong> with pH <strong>{ph}</strong> is already submitted. '
                         f'For job status visit <a href="https://fffold.biodata.ceitec.cz/results?ID={ID}" class="alert-link"'
                         f'target="_blank" rel="noreferrer">https://fffold.biodata.ceitec.cz/results?ID={ID}</a>.'), 'info')
            return render_template('index.html', status=server_status(), running=len(running), queued=len(queue))

        elif status == "unsubmitted":

            # download pdb
            response = requests.get(f'https://alphafold.ebi.ac.uk/files/AF-{code}-F1-model_v4.pdb')
            if response.status_code != 200:
                flash(Markup(f'The structure with code <strong>{code}</strong> '
                             f'is either not found in AlphaFold DB or the code is entered in the wrong format. '
                             f'UniProt code is allowed only in its short form (e.g. A0A1P8BEE7, B7ZW16). '
                             f'Other notations (e.g. A0A159JYF7_9DIPT, Q8WZ42-F2) are not supported. '
                             f'An alternative option is AlpfaFold DB Identifier (e.g. AF-L8BU87-F1).'), 'warning')
                return render_template('index.html', status=server_status(), running=len(running), queued=len(queue))
            data_dir = f'{root_dir}/calculated_structures/{ID}'
            calculation_time = len([line for line in response.text.split("\n") if line[:4] == "ATOM"])*2*0.14
            calculation_times[ID] = calculation_time
            os.mkdir(data_dir)
            with open(f'{data_dir}/{code}.pdb', 'w') as pdb:
                pdb.write(response.text)

            # create and submit job
            global optimizers
            optimizers = [optimizer for optimizer in optimizers if optimizer.is_alive()]
            queue.append(ID)
            if len(optimizers) < number_of_processes:
                optimizer = Process(target=optimize_structures)
                optimizer.start()
                optimizers.append(optimizer)
            return redirect(url_for('results', ID=ID))

    return render_template('index.html', status=server_status(), running=len(running), queued=len(queue))



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
        job_times = [finish_time-time() for finish_time in running.values()]
        for queued_job_ID in queue:
            job_times.append(calculation_times[queued_job_ID])
            if queued_job_ID == ID:
                break
        remaining_seconds = int(sum(job_times))
        remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
        remaining_hours, remaining_minutes = divmod(remaining_minutes, 60)
        time_string = ""
        if remaining_hours:
            time_string += f"{remaining_hours} hour{'s' if remaining_hours > 1 else ''}, "
        if remaining_minutes:
            time_string += f"{remaining_minutes} minute{'s' if remaining_minutes > 1 else ''} and "
        time_string += f"{remaining_seconds} seconds"
        status = f"Optimization is queued. Expected finish in {time_string}."
        return render_template('queued.html',
                               code=code,
                               ph=ph,
                               status=status)

    elif status == "running":
        remaining_seconds = int(running[ID]-time())
        if remaining_seconds < 0:
            status += " (The task takes an unusually long time. Wait, please.)"
        else:
            remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
            remaining_hours, remaining_minutes = divmod(remaining_minutes, 60)
            time_string = ""
            if remaining_hours:
                time_string += f"{remaining_hours} hour{'s' if remaining_hours>1 else ''}, "
            if remaining_minutes:
                time_string += f"{remaining_minutes} minute{'s' if remaining_minutes>1 else ''} and "
            time_string += f"{remaining_seconds} seconds"
            status = f"Optimization is running. Expected finish in {time_string}."
        return render_template('running.html',
                               code=code,
                               ph=ph,
                               status=status)

    return render_template('results.html',
                           ID=ID,
                           code=code,
                           ph=ph)


@application.route('/download_files')
def download_files():
    ID = request.args.get('ID')
    code, _ = ID.split("_")
    data_dir = f'{root_dir}/calculated_structures/{ID}'
    with zipfile.ZipFile(f'{data_dir}/{ID}.zip', 'w') as zip:
        zip.write(f'{data_dir}/optimization/optimized_PDB/{code}_added_H_optimized.pdb',f'{code}_optimized.pdb')
        zip.write(f'{data_dir}/optimization/optimized_CIF/{code}_added_H_optimized.cif', f'{code}_optimized.cif')
        zip.write(f'{data_dir}/optimization/inputed_PDB/{code}_added_H.pdb', f'{code}_original.pdb')
    return send_from_directory(data_dir, f'{ID}.zip', as_attachment=True)


@application.route('/optimized_structure/<ID>')
def get_optimized_structure(ID: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimization/optimized_CIF/{ID.split("_")[0]}_added_H_optimized.cif'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')


@application.route('/original_structure/<ID>')
def get_original_structure(ID: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimization/inputed_PDB/{ID.split("_")[0]}_added_H.pdb'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')


@application.route('/residues_logs/<ID>')
def get_residues_logs(ID: str):
    filepath = f'{root_dir}/calculated_structures/{ID}/optimization/residues.logs'
    return Response(open(filepath, 'r').read(), mimetype='text/plain')


@application.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
