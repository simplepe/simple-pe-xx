#! /usr/bin/env python

from argparse import ArgumentParser
import pycondor
import os

__authors__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


def command_line():
    """Define the command line arguments for `simple_pe_pipe`
    """
    from simple_pe_analysis import command_line as _analysis_command_line
    from simple_pe_filter import command_line as _filter_command_line
    parser = ArgumentParser(
        parents=[_analysis_command_line(), _filter_command_line()],
        conflict_handler='resolve'
    )
    remove = ["--peak_parameters", "--peak_snrs"]
    for action in parser._actions[::-1]:
        if action.option_strings[0] in remove:
            parser._handle_conflict_resolve(
                None, [(action.option_strings[0], action)]
            )
    parser.add_argument(
        "--accounting_group_user",
        help="Accounting group user to use for this workflow",
        required=True
    )
    parser.add_argument(
        "--accounting_group",
        help="Accounting group to use for this workflow",
        required=True
    )
    return parser


class Dag(object):
    """Base Dag object to handle the creation of the DAG

    Parameters
    ----------
    """
    def __init__(self, opts):
        self.opts = opts
        string = "%s/{}" % (opts.outdir)
        self.error = string.format("error")
        self.log = string.format("log")
        self.output = string.format("output")
        self.submit = string.format("submit")
        self.dagman = pycondor.Dagman(name="simple_pe", submit=self.submit)

    @property
    def bash_file(self):
        return os.path.join(self.submit, "bash_simple_pe.sh")

    @property
    def output_directories(self):
        dirs = {
            "error": self.error, "log": self.log, "output": self.output,
            "submit": self.submit
        }
        return dirs

    def build(self):
        """Build the pycondor dag
        """
        self.dagman.build()
        self.write_bash_script()
        print("Dag Generation complete.")
        print("To submit jobs to condor run:\n")
        print("$ condor_submit_dag {}\n".format(self.dagman.submit_file))
        print("To analyse jobs locally run:\n")
        print("$ bash {}\n".format(self.bash_file))

    def build_submit(self):
        """Submit the pycondor dag
        """
        self.dagman.build_submit()

    def write_bash_script(self):
        """Write a bash script containing all of the command lines used
        """
        with open(self.bash_file, "w") as f:
            f.write("#!/usr/bin/env bash\n\n")
            independent_jobs = []
            dependent_jobs = []
            for node in self.dagman.nodes:
                if len(node.parents):
                    dependent_jobs.append(node)
                else:
                    independent_jobs.append(node)
            for node in independent_jobs + dependent_jobs:
                f.write("# {}\n".format(node.name))
                f.write(
                    "# PARENTS {}\n".format(
                        " ".join([job.name for job in node.parents])
                    )
                )
                f.write(
                    "# CHILDREN {}\n".format(
                        " ".join([job.name for job in node.children])
                    )
                )
                job_str = "{} {}\n\n".format(node.executable, node.args[0].arg)
                job_str = job_str.replace("$(Cluster)", "0")
                job_str = job_str.replace("$(Process)", "0")
                f.write(job_str)


class Node(object):
    """Base node object to handle condor job creation

    Parameters
    ----------
    """
    def __init__(self, opts, dag):
        self.opts = opts
        self.dag = dag
        self.dagman = self.dag.dagman
        self._executable = self.get_executable("simple_pe_pipe")

    @staticmethod
    def get_executable(executable):
        """Find the path to the executable

        Parameters
        ----------
        executable: str
            the name of the executable you wish to find
        """
        from distutils.spawn import find_executable
        exe = find_executable(executable)
        if exe is None:
            return executable
        return exe

    @property
    def executable(self):
        return self._executable

    @property
    def request_memory(self):
        return "8 GB"

    @property
    def request_disk(self):
        return "1 GB"

    @property
    def request_cpus(self):
        return 1

    @property
    def getenv(self):
        return True

    @property
    def notification(self):
        return "never"

    @property
    def universe(self):
        return "vanilla"

    @property
    def extra_lines(self):
        return [
            "accounting_group = {}".format(self.opts.accounting_group),
            "accounting_group_user = {}".format(self.opts.accounting_group_user)
        ]

    def _format_arg_lists(self, string_args, dict_args, list_args):
        args = [
            [f"--{param}", str(getattr(self.opts, param))] for param in
            string_args if getattr(self.opts, param) is not None
        ]
        args += [
            [f"--{param}", " ".join([val for val in getattr(self.opts, param)])]
            for param in list_args if (getattr(self.opts, param) is not None)
            and len(getattr(self.opts, param))
        ]
        args += [
            [
                f"--{param}", " ".join([
                    f"{key}:{item}" for key, item in
                    getattr(self.opts, param).items()
                ])
            ] for param in dict_args if (getattr(self.opts, param) is not None)
            and len(getattr(self.opts, param))
        ]
        return args

    def add_parent(self, parent):
        """Add a parent to the node

        Parameters
        ----------
        parent:

        """
        self.job.add_parent(parent)

    def add_child(self, child):
        """Add a child to the node

        Parameters
        ----------
        child: varaha.condor.Node
            child node you wish to add to your job
        """
        self.job.add_child(child)

    def create_pycondor_job(self):
        self.job = pycondor.Job(
            name=self.job_name, executable=self.executable,
            submit=self.dag.submit, error=self.dag.error,
            log=self.dag.log, output=self.dag.output,
            request_memory=self.request_memory, request_disk=self.request_disk,
            request_cpus=self.request_cpus, getenv=self.getenv,
            universe=self.universe, extra_lines=self.extra_lines,
            dag=self.dagman, arguments=self.arguments,
            notification=self.notification
        )

class AnalysisNode(Node):
    """
    """
    job_name = "analysis"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("simple_pe_analysis")
        self.create_pycondor_job()

    @property
    def arguments(self):
        string_args = [
            "approximant", "f_low", "delta_f", "f_high", "minimum_data_length"
        ]
        dict_args = ["asd", "psd"]
        list_args = ["metric_directions", "precession_directions"]
        args = self._format_arg_lists(string_args, dict_args, list_args)
        args += [["--outdir", f"{self.opts.outdir}/output"]]
        args += [
            ["--peak_parameters", f"{self.opts.outdir}/output/peak_parameters.json"],
            ["--peak_snrs", f"{self.opts.outdir}/output/peak_snrs.json"]
        ]
        return " ".join([item for sublist in args for item in sublist])



class FilterNode(Node):
    """
    """
    job_name = "filter"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("simple_pe_filter")
        self.create_pycondor_job()

    @property
    def arguments(self):
        string_args = [
            "trigger_parameters", "approximant", "f_low", "f_high",
            "minimum_data_length"
        ]
        dict_args = ["strain", "asd", "psd"]
        args = self._format_arg_lists(string_args, dict_args, [])
        args += [["--outdir", f"{self.opts.outdir}/output"]]
        return " ".join([item for sublist in args for item in sublist])


def main(args=None):
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    MainDag = Dag(opts)
    FilterJob = FilterNode(opts, MainDag)
    AnalysisJob = AnalysisNode(opts, MainDag)
    FilterJob.add_child(AnalysisJob.job)
    MainDag.build()


if __name__ == "__main__":
    main()
