# shell script to analyse a zero-noise injection

# then define which waveform you wish to use
if [ -z ${WVF} ]; then
    WVF="IMRPhenomXPHM"
fi

# then run `simple_pe_pipe`
simple_pe_pipe --truth injection_params.json --strain H1:INJ:injection_params.json L1:INJ:injection_params.json --approximant ${WVF} --f_low 20 --f_high 2048 --accounting_group UNKNOWN --accounting_group_user albert.einstein --outdir ./outdir --trigger_parameters trigger_parameters.json --asd H1:aligo_O4high.txt L1:aligo_O4high.txt V1:avirgo_O4high_NEW.txt --metric_directions chirp_mass symmetric_mass_ratio chi_align --precession_directions symmetric_mass_ratio chi_align chi_p
# then run the output
bash ./outdir/submit/bash_simple_pe.sh
