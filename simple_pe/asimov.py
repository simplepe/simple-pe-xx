import importlib
import os
import configparser

from asimov.pipeline import Pipeline
from asimov import config
import htcondor
import yaml
from asimov.utils import set_directory

class AsimovPipeline(Pipeline):
    """
    An asimov pipeline for heron.
    """
    name = "simple_pe"
    config_template = importlib.resources.path("simple_pe", "asimov_template.yml")
    _pipeline_command = "simple_pe_pipe"

    def build_dag(self, dryrun=False):
        """
        Create a condor submission description.
        """
        name = self.production.name #meta['name']
        ini = self.production.event.repository.find_prods(name,
                                                          self.category)[0]

        user = "daniel.williams"
        
        command = f"""--strain H1:GWOSC:GW150914 L1:GWOSC:GW150914"""
        command += f"""--approximant {self.production.meta['waveform']['approximant']} """
        command += f"""--f_low 20 """
        command += f"""--f_high 2048 """
        command += f"""--accounting_group UNKNOWN """
        command += f"""--accounting_group_user {user} """
        command += f"""--outdir ./outdir """
        #command += f"""--trigger_parameters trigger_parameters.json """
        command += f"""--asd H1:GW150914_H1_asd.txt L1:GW150914_L1_asd.txt """
        #command += f"""--metric_directions ${METRIC_DIRECTIONS} """
        #command += f"""--precession_directions ${PREC_DIRECTIONS} """

        executable = f"{os.path.join(config.get('pipelines', 'environment'), 'bin', self._pipeline_command)}"

        print(executable, command)
        
        description = {
            "executable": executable,
            "arguments": command,
            "output": f"{name}.out",
            "error": f"{name}.err",
            "log": f"{name}.log",
            "request_gpus": 1,
            "batch_name": f"simple_pe/{name}",
        }

        job = htcondor.Submit(description)
        os.makedirs(self.production.rundir, exist_ok=True)
        with set_directory(self.production.rundir):
            with open(f"{name}.sub", "w") as subfile:
                subfile.write(job.__str__())

        with set_directory(self.production.rundir):
            try:
                schedulers = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd, config.get("condor", "scheduler"))
            except configparser.NoOptionError:
                schedulers = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd)
            schedd = htcondor.Schedd(schedulers)
            with schedd.transaction() as txn:
                cluster_id = job.queue(txn)

        self.clusterid = cluster_id

    def submit_dag(self, dryrun=False):
        return self.clusterid
    
def submit_description():
    schedd = htcondor.Schedd(schedulers)
    with schedd.transaction() as txn:   
        cluster_id = job.queue(txn)
    return cluster_id

