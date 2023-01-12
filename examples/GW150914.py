from subprocess import check_output
import json

# download ASD for H1 and L1 from GWOSC
base = "https://www.gw-openscience.org/GW150914data/P1500238/"
_ = check_output(f"curl {base}/H1-GDS-CALIB_STRAIN.txt -o ./GW150914_H1_asd.txt", shell=True)
_ = check_output(f"curl {base}/L1-GDS-CALIB_STRAIN.txt -o ./GW150914_L1_asd.txt", shell=True)

# write injection parameters to json file
with open("trigger_parameters.json", "w") as f:
    injection_params = {
        "mass_1": 36.8,
        "mass_2": 32.0,
        "spin_1z": -0.62,
        "spin_2z": 0.47,
        "chi_p2": 0.0625,
        "distance": 1.0,
        "inclination": 2.4,
        "time": 1126259462.41,
        "ra": 2.2,
        "dec": -1.2,
        "psi": 0.7
    }
    json.dump(injection_params, f)

# run simple_pe_pipe
arguments = [
    "--strain", "H1:GWOSC:GW150914", "L1:GWOSC:GW150914",
    "--approximant", opts.approximant, "--f_low", 20,
    "--f_high", 2048, "--accounting_group", "UNKNOWN",
    "--accounting_group_user", "albert.einstein",
    "--outdir", "./outdir", "--trigger_parameters"

curl https://www.gw-openscience.org/GW150914data/P1500238/H1-GDS-CALIB_STRAIN.txt -o ./GW150914_H1_asd.txt
    - curl https://www.gw-openscience.org/GW150914data/P1500238/L1-GDS-CALIB_STRAIN.txt -o ./GW150914_L1_asd.txt
    - "echo '{\"mass_1\": 36.8, \"mass_2\": 32.0, \"spin_1z\": -0.62, \"spin_2z\": 0.47, \"chi_p2\": 0.0625, \"distance\": 1.0, \"inclination\": 2.4, \"time\": 1126259462.41, \"ra\": 2.2, \"dec\": -1.2, \"psi\": 0.7}' > trigger_parameters.json"
    - simple_pe_pipe --strain H1:GWOSC:GW150914 L1:GWOSC:GW150914 --approximant ${WVF} --f_low 20 --f_high 2048 --accounting_group UNKNOWN --accounting_group_user albert.einstein --outdir ./outdir --trigger_parameters trigger_parameters.json --asd H1:GW150914_H1_asd.txt L1:GW150914_L1_asd.txt --metric_directions ${METRIC_DIRECTIONS} --precession_directions ${PREC_DIRECTIONS}
    - bash ./outdir/submit/bash_simple_pe.sh
