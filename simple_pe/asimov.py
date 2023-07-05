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

        print("Config template", self.config_template)
        
        if self.production.event.repository:
            ini = self.production.event.repository.find_prods(
                self.production.name, self.category
            )[0]
            ini = os.path.join(cwd, ini)
        else:
            ini = f"{self.production.name}.ini"

        
        command = f"""{ini}"""

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

