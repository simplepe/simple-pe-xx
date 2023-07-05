import importlib
import os
import configparser
import subprocess
import glob
from asimov.pipeline import Pipeline, PipelineException
from asimov import config
import htcondor
import yaml
from asimov.utils import set_directory

class AsimovPipeline(Pipeline):
    """
    An asimov pipeline for heron.
    """
    name = "simple_pe"
    with importlib.resources.path("simple_pe", "asimov_template.yml") as template_file:
        config_template = template_file

    #config_template = importlib.resources.path("simple_pe", "asimov_template.yml")
    _pipeline_command = "simple_pe_pipe"

    def build_dag(self, dryrun=False):
        """
        Create a condor submission description.
        """
        name = self.production.name #meta['name']
        ini = self.production.event.repository.find_prods(name,
                                                          self.category)[0]
        cwd = os.getcwd()
        
        if self.production.event.repository:
            ini = self.production.event.repository.find_prods(
                self.production.name, self.category
            )[0]
            ini = os.path.join(cwd, ini)
        else:
            ini = f"{self.production.name}.ini"

        
        command = f"""{ini}"""

        executable = f"{os.path.join(config.get('pipelines', 'environment'), 'bin', self._pipeline_command)}"

        if dryrun:
            print(" ".join([executable, command]))
        else:
            if not os.path.exists(self.production.rundir):
                os.makedirs(self.production.rundir)
            
            with set_directory(self.production.rundir): 
            
                self.logger.info(" ".join([executable, command]))
                pipe = subprocess.Popen(
                    [executable, command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                out, err = pipe.communicate()
                self.logger.info(out)

                if err or "DAG generation complete, to submit jobs" not in str(out):
                    self.production.status = "stuck"
                    self.logger.error(err)
                    raise PipelineException(
                        f"DAG file could not be created.\n{command}\n{out}\n\n{err}",
                        production=self.production.name,
                    )
                else:
                    time.sleep(10)
                    return None #PipelineLogger(message=out, production=self.production.name)

    def submit_dag(self, dryrun=False):
        cwd = os.getcwd()
        self.logger.info(f"Working in {cwd}")

        self.before_submit()

        try:
            # to do: Check that this is the correct name of the output DAG file for billby (it
            # probably isn't)
            if "job label" in self.production.meta:
                job_label = self.production.meta["job label"]
            else:
                job_label = self.production.name
            dag_filename = glob.glob(os.path.join(self.production.rundir, "submit", "*.submit"))[0]
            command = [
                # "ssh", f"{config.get('scheduler', 'server')}",
                "condor_submit_dag",
                "-batch-name",
                f"simple_pe/{self.production.event.name}/{self.production.name}",
                os.path.join(self.production.rundir, "submit", dag_filename),
            ]

            if dryrun:
                print(" ".join(command))
            else:

                # with set_directory(self.production.rundir):
                self.logger.info(f"Working in {os.getcwd()}")

                dagman = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )

                self.logger.info(" ".join(command))

                stdout, stderr = dagman.communicate()

                if "submitted to cluster" in str(stdout):
                    cluster = re.search(
                        r"submitted to cluster ([\d]+)", str(stdout)
                    ).groups()[0]
                    self.logger.info(
                        f"Submitted successfully. Running with job id {int(cluster)}"
                    )
                    self.production.status = "running"
                    self.production.job_id = int(cluster)
                    return cluster, PipelineLogger(stdout)
                else:
                    self.logger.error("Could not submit the job to the cluster")
                    self.logger.info(stdout)
                    self.logger.error(stderr)

                    raise PipelineException(
                        "The DAG file could not be submitted.",
                    )

        except FileNotFoundError as error:
            self.logger.exception(error)
            raise PipelineException(
                "It looks like condor isn't installed on this system.\n"
                f"""I wanted to run {" ".join(command)}."""
            ) from error

