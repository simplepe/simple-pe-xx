# shell script to analyse a zero-noise injection

# then define which waveform you wish to use
if [ -z ${WVF} ]; then
    WVF="IMRPhenomXPHM"
fi

# then run `simple_pe_pipe`
simple_pe_pipe --truth injection_params.json --strain H1:INJ:injection_params.json L1:INJ:injection_params.json --approximant ${WVF} --f_low 20 --f_high 2048 --accounting_group UNKNOWN --accounting_group_user albert.einstein --outdir ./outdir --trigger_parameters trigger_parameters.json --asd H1:../GW150914/GW150914_H1_asd.txt L1:../GW150914/GW150914_L1_asd.txt --metric_directions chirp_mass symmetric_mass_ratio chi_align --precession_directions symmetric_mass_ratio chi_align chi_p
# then run the output
bash ./outdir/submit/bash_simple_pe.sh
